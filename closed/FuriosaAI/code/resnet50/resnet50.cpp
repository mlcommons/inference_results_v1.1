#include <cassert>
#include <sstream>
#include <condition_variable>
#include <deque>
#include <iostream>
#include <map>
#include <mutex>
#include <thread>
#include <vector>
#include <fstream>
#include <streambuf>
#include <atomic>

#include <string.h>

#include "loadgen.h"
#include "query_sample_library.h"
#include "system_under_test.h"
#include "test_settings.h"

#include "../common/nux.h"
#include <stdlib.h>
#include "../common/unlower.h"

using namespace std::string_literals;

// #define ASYNC
// #define PROFILE
// #define DEBUG
#define POST_PROCESS

#ifdef PROFILE
std::string PROFILE_RECORD_INFERENCE = "Inference";
std::string PROFILE_RECORD_PREPARE_SUBMISSION = "PrepareSubmission";
std::string PROFILE_RECORD_POST_PROCESS = "PostProcess";
std::string PROFILE_RECORD_OUT_OF_TIMED = "OutOfTimed";
std::string build_profile_record(std::string record_name, std::string annotation) {
    return record_name + "(" + annotation + ")";
}
#endif

std::string DATA_PATH = "../preprocessed/imagenet-golden/raw/";
std::string VAL_MAP_PATH = "../common/val_map.txt";

const int IMAGE_SIZE = (224*224*3);
int INPUT_SIZE = IMAGE_SIZE;
int OUTPUT_SIZE = 1001;
std::vector<std::unique_ptr<unsigned char[]>> images;

std::optional<LoweringInfo> input_lowering;
std::optional<LoweringInfo> output_lowering;

std::optional<LoweringInfo> TensorDescToLoweringInfo(nux_tensor_desc_t desc, int unlowered_channel, int unlowered_height, int unlowered_width) {
    int dim = nux_tensor_dim_num(desc);
    if (dim != 6) 
        return {};
    if (nux_tensor_axis(desc, 0) != Axis::axis_batch)
        return {};

    if (nux_tensor_axis(desc, 1) == Axis::axis_height_outer &&
        nux_tensor_axis(desc, 2) == Axis::axis_channel_outer &&
        nux_tensor_axis(desc, 3) == Axis::axis_height 
        ) {
        if (nux_tensor_axis(desc, 4) == Axis::axis_channel &&
            nux_tensor_axis(desc, 5) == Axis::axis_width) {
            return DynamicHCHCW(
                nux_tensor_dim(desc, 1),
                nux_tensor_dim(desc, 2),
                nux_tensor_dim(desc, 3),
                nux_tensor_dim(desc, 4),
                nux_tensor_dim(desc, 5),
                unlowered_channel,
                unlowered_height,
                unlowered_width
            );
        }
        if (nux_tensor_axis(desc, 4) == Axis::axis_width &&
            nux_tensor_axis(desc, 5) == Axis::axis_channel
            ) {
            return DynamicHCHWC(
                nux_tensor_dim(desc, 1),
                nux_tensor_dim(desc, 2),
                nux_tensor_dim(desc, 3),
                nux_tensor_dim(desc, 4),
                nux_tensor_dim(desc, 5),
                unlowered_channel,
                unlowered_height,
                unlowered_width
            );
        }
    }

    return {};
}

class QSL : public mlperf::QuerySampleLibrary {
 public:
  QSL(int sampleSize = 50000) 
    : mSampleSize(sampleSize) {
        std::ifstream val_file(VAL_MAP_PATH.c_str());

        std::string input_filename;

        std::cout << "sample size: " << std::to_string(sampleSize) << std::endl;
        int answer;
        while(val_file >> input_filename >> answer) {
            mItems.emplace_back(input_filename.substr(0, input_filename.size() - 5) + ".JPEG.raw", answer);
        }
        images.resize(mItems.size());
    };
  ~QSL() override{};
  const std::string& Name() const override { return mName; }
  size_t TotalSampleCount() override { return mSampleSize; }
  size_t PerformanceSampleCount() override { 
      std::cout << "PerformanceSampleCount" << 10240 << '\n';
      return 10240; 
  }
  void LoadSamplesToRam(
          const std::vector<mlperf::QuerySampleIndex>& samples) override {
      for(auto index : samples) {
          std::string filename = DATA_PATH + mItems[index].first;
          std::ifstream inf(filename.c_str(), std::ios::binary);

          if (input_lowering) {
              std::vector<char> buffer(IMAGE_SIZE);
              inf.read((char*)&buffer[0], 224*224*3);
              std::visit([&](auto info){ 
                  INPUT_SIZE = info.lowered_size();
                  images[index].reset((unsigned char*)std::aligned_alloc(64, info.lowered_size()));
                  info.for_each([&](int c, int co, int ci, int h, int ho, int hi, int w) {
                      images[index][info.index(co, ci, ho, hi, w)] = buffer[c*224*224+h*224+w];
                  });
              }, *input_lowering);

          } else {
              images[index].reset((unsigned char*)std::aligned_alloc(64, IMAGE_SIZE));
              inf.read((char*)&images[index][0], 224*224*3);
          }
      }
  }
  void UnloadSamplesFromRam(
      const std::vector<mlperf::QuerySampleIndex>& samples) override {
    for(auto index : samples) {
        //std::cout << "UNLOAD " << index << '\n';
        images[index].reset();
    }
  }

 private:
  std::string mName{"FuriosaAI-QSL"};
  int mSampleSize;
  std::vector<std::pair<std::string, int>> mItems;
};

class Resnet50Reporter {
    protected:
  void post_inference(unsigned char* buffer, 
          mlperf::QuerySampleResponse& response) {

        float max_index = 0;
        char* data_arr = (char*)buffer;
        if (output_lowering) {
            std::visit([&](auto info){
                for(int i = 1; i < 1001; i ++) {
                    if (data_arr[info.index((int)max_index, 0, 0)] < data_arr[info.index(i, 0, 0)]) {
                        max_index = i;
                    }
                }
            }, *output_lowering);

        } else {
            for(int i = 1; i < 1001; i ++) {
                if (data_arr[(int)max_index] < data_arr[i]) {
                    max_index = i;
                }
            }
        }

        *(float*)response.data = max_index;
  }
};


class FuriosaBasicSUT : public mlperf::SystemUnderTest, Resnet50Reporter {
 public:
    nux_session_t mSession;
    nux_completion_queue_t mQueue;
    nux_session_option_t mSessionOption;
    nux_tensor_array_t mInputs;
    nux_tensor_array_t mOutputs;
    std::atomic<int> mIssued;
    std::atomic<int> mCompleted;

  FuriosaBasicSUT() 
  {
    // Start with some large value so that we don't reallocate memory.
    initResponse(1);

    std::ifstream inf("mlcommons_resnet50_v1.5_int8.enf", std::ios::binary);

    mSessionOption = nux_session_option_create();
    nux_session_option_set_device(mSessionOption, "npu0pe0-1");

    std::vector<char> model_buf((std::istreambuf_iterator<char>(inf)),
                     std::istreambuf_iterator<char>());

    {
        auto err = nux_session_create((nux_buffer_t)model_buf.data(), model_buf.size(), mSessionOption, &mSession);
        if (err != nux_error_t_success) {
            std::cerr << "SUT:nux async session create error: " << err << '\n';
            exit(-1);
        }
    }

    nux_model_t model = nux_session_get_model(mSession);
    mInputs = nux_tensor_array_create_inputs(model);
    mOutputs = nux_tensor_array_allocate_outputs(model);


    auto input_desc = nux_input_desc(model, 0);
    input_lowering = TensorDescToLoweringInfo(input_desc, 3, 224, 224);
    auto output_desc =  nux_output_desc(model, 0);
    output_lowering = TensorDescToLoweringInfo(output_desc, 1001, 1, 1);
    if (output_lowering)
        std::visit([](auto info){OUTPUT_SIZE = info.lowered_size();}, *output_lowering);
  }

  std::vector<std::thread> mCompleteThreads;

  ~FuriosaBasicSUT() override {
      nux_session_destroy(mSession);
  }

  const std::string& Name() const override { return mName; }
  void IssueQuery(const std::vector<mlperf::QuerySample>& samples) override {
    int n = samples.size();
    if (n > mResponses.size()) {
      std::cerr << "Warning: reallocating response buffer in BasicSUT. Maybe "
                   "you should initResponse with larger value!?"
                << std::endl;
      initResponse(samples.size());
    }
    for (int i = 0; i < n; i++) {
        mResponses[i].id = samples[i].id;
        auto tensor0 = nux_tensor_array_get(mInputs, 0);
        auto* data = images[samples[i].index].get();
        nux_error_t err;
        //std::cout << "Issuing: " << samples[i].index << ' ' << i << '/' << n << ' ' << data.size() << '\n';

        // Use allocated aligned buffer
        tensor_set_buffer(tensor0, (nux_buffer_t)data, INPUT_SIZE, nullptr);
        //auto err = tensor_set_buffer(tensor0, (nux_buffer_t)data.data(), data.size(), nullptr);

        err = nux_session_run(mSession, mInputs, mOutputs);
        if (err != 0) {
            std::cout << "Error: " << err << '\n';
        }

        auto result = nux_tensor_array_get(mOutputs, 0);
        nux_buffer_t buffer;
        nux_buffer_len_t len;
        tensor_get_buffer(result, &buffer, &len);

        post_inference(buffer, mResponses[0]);

        mlperf::QuerySamplesComplete(mResponses.data(), 1);
    }
  }

  void FlushQueries() override {}
  void ReportLatencyResults(
      const std::vector<mlperf::QuerySampleLatency>& latencies_ns) override{};

 private:
  void initResponse(int size) {
    mResponses.resize(size,
                      {0, reinterpret_cast<uintptr_t>(&mBuf), sizeof(mBuf)});
  }
  float mBuf{1};
  std::string mName{"FuriosaAI-BasicSUT"};
  std::vector<mlperf::QuerySampleResponse> mResponses;
};


class FuriosaQueueSUT : public mlperf::SystemUnderTest, Resnet50Reporter {
 public:
 std::vector<nux_session_t> mSessions;
  FuriosaQueueSUT(int numCompleteThreads, int maxSize) {

    std::ifstream inf("mlcommons_resnet50_v1.5_int8_batch8.enf", std::ios::binary);
    std::vector<char> model_buf((std::istreambuf_iterator<char>(inf)),
                     std::istreambuf_iterator<char>());

    for(int npu_id = 0; npu_id < numCompleteThreads / 2; npu_id++) {
        for(int pe_id = 0; pe_id < 2; pe_id++) {
            std::stringstream ostr;
            ostr << "npu" << npu_id << "pe" << pe_id;

            auto sessionOption = nux_session_option_create();
            nux_session_option_set_device(sessionOption, ostr.str().c_str());

            nux_session_t session;
            auto err = nux_session_create((nux_buffer_t)model_buf.data(), model_buf.size(), sessionOption, &session);
            if (err != nux_error_t_success) {
                std::cerr << "SUT:nux session create error: " << err << '\n';
                exit(-1);
            }

            mSessions.push_back(session);


            if (npu_id == 0 && pe_id == 0) {
                nux_model_t model = nux_session_get_model(session);

                auto input_desc = nux_input_desc(model, 0);
                input_lowering = TensorDescToLoweringInfo(input_desc, 3, 224, 224);
                auto output_desc =  nux_output_desc(model, 0);
                output_lowering = TensorDescToLoweringInfo(output_desc, 1001, 1, 1);
                if (output_lowering)
                    std::visit([](auto info){OUTPUT_SIZE = info.lowered_size();}, *output_lowering);
            }
        }
    }
    initResponse(numCompleteThreads, maxSize);
    for (int i = 0; i < numCompleteThreads; i++) {
      mThreads.emplace_back(&FuriosaQueueSUT::CompleteThread, this, i);
    }
  }
  ~FuriosaQueueSUT() override {
    {
      std::unique_lock<std::mutex> lck(mMtx);
      mDone = true;
      mCondVar.notify_all();
    }
    for (auto& thread : mThreads) {
      thread.join();
    }
  }
  const std::string& Name() const override { return mName; }
  void IssueQuery(const std::vector<mlperf::QuerySample>& samples) override {
      {
          std::unique_lock<std::mutex> lck(mMtx);
          mWorkQueue = &samples;
          mQueueTail = 0;
      }
    mCondVar.notify_one();
  }
  void FlushQueries() override {}
  void ReportLatencyResults(
      const std::vector<mlperf::QuerySampleLatency>& latencies_ns) override{};

 private:
  void CompleteThread(int threadIdx) {
    auto& inputs = mInputs[threadIdx];
    auto& responses = mResponses[threadIdx];
    auto& session = mSessions[threadIdx];

    nux_model_t model = nux_session_get_model(session);

    auto input_tensor = nux_tensor_array_create_inputs(model);
    auto output_tensor = nux_tensor_array_allocate_outputs(model);

    size_t maxSize{responses.size()};
    size_t actualSize{0};
    while (true) {
      {
        std::unique_lock<std::mutex> lck(mMtx);
        mCondVar.wait(lck, [&]() { return mWorkQueue && mWorkQueue->size() != mQueueTail || mDone; });

        if (mDone) {
          break;
        }

        auto* requests = &(*mWorkQueue)[mQueueTail];
        actualSize = std::min(maxSize, mWorkQueue->size() - mQueueTail);
        mQueueTail += actualSize;
        for (int i = 0; i < actualSize; i++) {
          responses[i].id = requests[i].id;
        }
        if (mWorkQueue->size() == mQueueTail) {
            mWorkQueue = nullptr;
        }
        lck.unlock();
        mCondVar.notify_one();
        for(int i = 0; i < actualSize; i ++) {
            int index = requests[i].index;
            auto* data = images[index].get();
            memcpy(&inputs[i*INPUT_SIZE] , data, INPUT_SIZE);
        }

        auto tensor0 = nux_tensor_array_get(input_tensor, 0);
        tensor_set_buffer(tensor0, (nux_buffer_t)&inputs[0], maxSize * INPUT_SIZE, nullptr);
        auto err = nux_session_run(session, input_tensor, output_tensor);
        if (err != 0) {
            std::cout << "Error: " << err << '\n';
        }

        auto result = nux_tensor_array_get(output_tensor, 0);
        nux_buffer_t buffer;
        nux_buffer_len_t len;
        tensor_get_buffer(result, &buffer, &len);
        for(int i = 0; i < actualSize; i ++)
            post_inference((unsigned char*)buffer + OUTPUT_SIZE * i, responses[i]);

      }
      mlperf::QuerySamplesComplete(responses.data(), actualSize);
    }
  }
  void initResponse(int numCompleteThreads, int size) {
      mBuf.resize(numCompleteThreads*size);
      mInputs.resize(numCompleteThreads);
      for(int i = 0; i < numCompleteThreads; i ++)
          mInputs[i].reset((unsigned char*)std::aligned_alloc(64, INPUT_SIZE*size));
      mResponses.resize(numCompleteThreads);
      for(int i = 0; i < numCompleteThreads; i ++) {
          for(int j = 0; j < size; j ++) {
              mResponses[i].emplace_back(mlperf::QuerySampleResponse{0, reinterpret_cast<uintptr_t>(&mBuf[i*size+j]), sizeof(float)});
          }
      }
  }
  std::vector<float> mBuf;
  std::string mName{"FuriosaQueueSUT"};
  std::vector<std::vector<mlperf::QuerySampleResponse>> mResponses;
  std::vector<std::unique_ptr<unsigned char[]>> mInputs;
  std::vector<std::thread> mThreads;
  const std::vector<mlperf::QuerySample>* mWorkQueue{nullptr};
  size_t mQueueTail;
  std::mutex mMtx;
  std::condition_variable mCondVar;
  bool mDone{false};
};

struct Context {
    mlperf::ResponseId *ids;
    nux_tensor_array_t input;
    nux_tensor_array_t output;
    unsigned char *buffer;
    int len;
};

class StreamSession : public Resnet50Reporter {
public:
    // Nux Session
    nux_async_session_t mSession;
    nux_completion_queue_t mQueue;
    nux_handle_t mNux;
    nux_model_t mModel;
    // SessionWarpper options
    nux_session_option_t mSessionOption;
    // IOs
    nux_tensor_array_t mInput;
    nux_tensor_array_t mOutput;
    mlperf::ResponseId *mIds;
    unsigned char *mBuffer;
    Context *mContext;
    // Compltion thread
    std::thread mCompletionThread;
    int mModelBatch;

    const std::string& Name() const { return mName; }

    StreamSession(
		int numWorkers = 12, 
		int modelBatch = 1,
		std::string deviceName = "npu0pe0-1",
		std::string source = "") {
#ifdef DEBUG
        std::cout << "Constructing StreamSession(" << numWorkers << ")" << std::endl;
#endif
        mModelBatch = modelBatch;
        mSessionOption = nux_session_option_create();
        nux_session_option_set_worker_num(mSessionOption, numWorkers);
		nux_session_option_set_device(mSessionOption, deviceName.c_str());
		nux_session_option_set_input_queue_size(mSessionOption, 8);
		nux_session_option_set_output_queue_size(mSessionOption, 256);

#ifdef DEBUG
        std::cout << "Prepare Nux session with device name: " << deviceName << std::endl;
#endif
        std::ifstream inf(source, std::ios::binary);
        std::vector<char> model_buf((std::istreambuf_iterator<char>(inf)),
                                    std::istreambuf_iterator<char>());
        auto err = nux_async_session_create(
                (nux_buffer_t)model_buf.data(),
                model_buf.size(),
                mSessionOption,
                &mSession,
                &mQueue);
#ifdef DEBUG
        std::cout << "create async session done: " << deviceName << std::endl;
#endif
        if (err != nux_error_t_success) {
            std::cerr << "SUT:nux async session create error: " << err << std::endl;
            exit(-1);
        }
        mModel = nux_async_session_get_model(mSession);
        nux_async_session_get_nux_handle(mSession, &mNux);

#ifdef DEBUG
        std::cout << "Prepare first submission" << std::endl;
#endif
        mInput = nux_tensor_array_create_inputs(mModel);
        mOutput = nux_tensor_array_allocate_outputs(mModel);
        mContext = (Context *) malloc(sizeof(Context));
        mIds = (mlperf::ResponseId *) malloc(sizeof(mlperf::ResponseId) * mModelBatch);
        mBuffer = (unsigned char*) std::aligned_alloc(64, INPUT_SIZE * mModelBatch);

        auto input_desc = nux_input_desc(mModel, 0);
        input_lowering = TensorDescToLoweringInfo(input_desc, 3, 224, 224);
        auto output_desc =  nux_output_desc(mModel, 0);
        output_lowering = TensorDescToLoweringInfo(output_desc, 1001, 1, 1);
        if (output_lowering)
            std::visit([](auto info){OUTPUT_SIZE = info.lowered_size();}, *output_lowering);

#ifdef DEBUG
        std::cout << "Run completion thread" << std::endl;
#endif
        mCompletionThread = std::thread([this] { CompleteThread(); });
    }

    ~StreamSession() {
        nux_tensor_array_destroy(mInput);
        nux_tensor_array_destroy(mOutput);
        free(mContext);
        free(mIds);
        nux_async_session_destroy(mSession);
        nux_completion_queue_destroy(mQueue);
        mCompletionThread.join();
    }

    void SubmitQuery(std::vector<mlperf::QuerySample> samples, int begin, int len) {
#ifdef PROFILE
        nux_profiler_record_begin(mNux, build_profile_record(PROFILE_RECORD_INFERENCE, std::to_string(samples[begin].id)).c_str());
        nux_profiler_record_begin(mNux, build_profile_record(PROFILE_RECORD_PREPARE_SUBMISSION, std::to_string(samples[begin].id)).c_str());
#endif
        if (len == 1) {
            mIds[0] = samples[begin].id;
            auto tensor = nux_tensor_array_get(mInput, 0);
            auto* data = images[samples[begin].index].get();
            auto err = tensor_set_buffer(tensor, (nux_buffer_t) data, INPUT_SIZE, nullptr);
            if (err != 0) {
                std::cout << "Error: " << err << '\n';
            }
        } else {
            for(int i = 0; i < len ; i ++) {
                mIds[i] = samples[begin + i].id;
                auto* data = images[samples[begin + i].index].get();
                memcpy(&mBuffer[i*INPUT_SIZE] , data, INPUT_SIZE);
            }
            auto tensor = nux_tensor_array_get(mInput, 0);
            auto err = tensor_set_buffer(tensor, (nux_buffer_t)&mBuffer[0], mModelBatch * INPUT_SIZE, nullptr);
            if (err != 0) {
                std::cout << "Error: " << err << '\n';
            }
        }
        mContext->ids = mIds;
        mContext->input = mInput;
        mContext->output = mOutput;
        mContext->buffer = mBuffer;
        mContext->len = len;
#ifdef PROFILE
        nux_profiler_record_end(mNux, build_profile_record(PROFILE_RECORD_PREPARE_SUBMISSION, std::to_string(samples[begin].id)).c_str());
#endif
#ifdef DEBUG
        std::cout 
            << "Submit ctx addr: " << (void *) mContext 
            << ", ctx->output: " << (void *) mContext->output 
            << ", ctx->input: " << (void *) mContext->input 
            << std::endl;
#endif

        auto err = nux_async_session_run(mSession, (nux_context_t) mContext, mInput, mOutput);
        if (err != 0) {
            std::cout << "nux session run error: " << err << '\n';
        }

#ifdef PROFILE
        nux_profiler_record_begin(mNux, build_profile_record(PROFILE_RECORD_OUT_OF_TIMED, "alloc-for-next-" + std::to_string(samples[begin].id)).c_str());
#endif
        mInput = nux_tensor_array_create_inputs(mModel);
        mOutput = nux_tensor_array_allocate_outputs(mModel);
        mContext = (Context *) malloc(sizeof(Context));
        mIds = (mlperf::ResponseId *) malloc(sizeof(mlperf::ResponseId) * mModelBatch);
        mBuffer = (unsigned char*) std::aligned_alloc(64, INPUT_SIZE * mModelBatch);
#ifdef PROFILE
        nux_profiler_record_end(mNux, build_profile_record(PROFILE_RECORD_OUT_OF_TIMED, "alloc-for-next-" + std::to_string(samples[begin].id)).c_str());
#endif
    }

    void CompleteThread() {
        std::cout << "Completion Thread has been launched" << '\n';
        nux_context_t context;
        nux_tensor_array_t output;
        float tempBuf[mModelBatch];
        std::vector<mlperf::QuerySampleResponse> responses;
		for (int i = 0 ; i < mModelBatch ; i++) {
			responses.push_back(mlperf::QuerySampleResponse {0, reinterpret_cast<uintptr_t>(&tempBuf[i]), sizeof(float)});
		}
        enum nux_error_t error;
        while (nux_completion_queue_next(mQueue,
                                         &context,
                                         &error)) {
            Context* ctx = (Context*) context;
#ifdef DEBUG
            std::cout 
                << "Recv ctx addr: " << (void *) ctx
                << ", ctx->output: " << (void *) ctx->output 
                << ", ctx->input: " << (void *) ctx->input 
                << std::endl;
#endif
#ifdef PROFILE
            nux_profiler_record_begin(mNux, build_profile_record(PROFILE_RECORD_POST_PROCESS, std::to_string(ctx->ids[0])).c_str());
#endif
            auto result = nux_tensor_array_get(ctx->output, 0);
            nux_buffer_t buffer;
            nux_buffer_len_t buffer_len;
            tensor_get_buffer(result, &buffer, &buffer_len);
            for (int i = 0; i < ctx->len ; i ++) {
                responses[i].id = ctx->ids[i];
                responses[i].size = sizeof(float);
#ifdef POST_PROCESS
                post_inference((unsigned char*)buffer + OUTPUT_SIZE * i, responses[i]);
#else
                responses[i].data = (uintptr_t)&buffer + OUTPUT_SIZE * i;
#endif
            }

#ifdef PROFILE
            nux_profiler_record_end(mNux, build_profile_record(PROFILE_RECORD_POST_PROCESS, std::to_string(ctx->ids[0])).c_str());
#endif
            mlperf::QuerySamplesComplete(responses.data(), ctx->len);
#ifdef PROFILE
            nux_profiler_record_end(mNux, build_profile_record(PROFILE_RECORD_INFERENCE, std::to_string(ctx->ids[0])).c_str());
#endif

#ifdef PROFILE
            nux_profiler_record_begin(mNux, build_profile_record(PROFILE_RECORD_OUT_OF_TIMED, "dealloc-" + std::to_string(ctx->ids[0])).c_str());
#endif
            nux_tensor_array_destroy(ctx->input);
            nux_tensor_array_destroy(ctx->output);
            free(ctx->ids);
            free(ctx->buffer);
            free(ctx);
#ifdef PROFILE
            nux_profiler_record_end(mNux, build_profile_record(PROFILE_RECORD_OUT_OF_TIMED, "dealloc-" + std::to_string(ctx->ids[0])).c_str());
#endif
        }
    }
private:
  std::string mName{"FuriosaAI-StreamSession-SUT"};
};

class BlockingSession : public Resnet50Reporter {
public:
    // Nux Session
    nux_session_t mSession;
    nux_handle_t mNux;
    nux_model_t mModel;
    // SessionWarpper options
    nux_session_option_t mSessionOption;
    // IOs
    nux_tensor_array_t mInput;
    nux_tensor_array_t mOutput;
    std::vector<mlperf::QuerySampleResponse> mResponses;

    const std::string& Name() const { return mName; }

    BlockingSession() {
#ifdef DEBUG
        std::cout << "Constructing BlockingSession" << std::endl;
#endif
        mSessionOption = nux_session_option_create();
		nux_session_option_set_device(mSessionOption, "npu0pe0-1");

#ifdef DEBUG
        std::cout << "Prepare Nux session" << std::endl;
#endif
        std::ifstream inf("mlcommons_resnet50_v1.5_int8.enf", std::ios::binary);
        std::vector<char> model_buf((std::istreambuf_iterator<char>(inf)),
                                    std::istreambuf_iterator<char>());
        auto err = nux_session_create(
                (nux_buffer_t)model_buf.data(),
                model_buf.size(),
                mSessionOption,
                &mSession);
        if (err != nux_error_t_success) {
            std::cerr << "SUT:nux session create error: " << err << std::endl;
            exit(-1);
        }
        mModel = nux_session_get_model(mSession);
        nux_session_get_nux_handle(mSession, &mNux);

        auto input_desc = nux_input_desc(mModel, 0);
        input_lowering = TensorDescToLoweringInfo(input_desc, 3, 224, 224);
        auto output_desc =  nux_output_desc(mModel, 0);
        output_lowering = TensorDescToLoweringInfo(output_desc, 1001, 1, 1);
        if (output_lowering)
            std::visit([](auto info){OUTPUT_SIZE = info.lowered_size();}, *output_lowering);

#ifdef DEBUG
        std::cout << "Prepare first submission" << std::endl;
#endif
        mInput = nux_tensor_array_create_inputs(mModel);
        mOutput = nux_tensor_array_allocate_outputs(mModel);
        int tempBuf{0};
        mResponses.resize(1, {0, reinterpret_cast<uintptr_t>(&tempBuf), sizeof(int)});
    }

    ~BlockingSession() {
        nux_tensor_array_destroy(mInput);
        nux_tensor_array_destroy(mOutput);
        nux_session_destroy(mSession);
    }

    void Run(mlperf::QuerySample sample) {
#ifdef DEBUG
        std::cout << "run sample: " << std::to_string(sample.id) << std::endl;
#endif
#ifdef PROFILE
        nux_profiler_record_begin(mNux, build_profile_record(PROFILE_RECORD_INFERENCE, std::to_string(sample.id)).c_str());
        nux_profiler_record_begin(mNux, build_profile_record(PROFILE_RECORD_PREPARE_SUBMISSION, std::to_string(sample.id)).c_str());
#endif

        auto tensor = nux_tensor_array_get(mInput, 0);
        auto* data = images[sample.index].get();
        auto err = tensor_set_buffer(tensor, (nux_buffer_t) data, INPUT_SIZE, nullptr);
#ifdef PROFILE
        nux_profiler_record_end(mNux, build_profile_record(PROFILE_RECORD_PREPARE_SUBMISSION, std::to_string(sample.id)).c_str());
#endif
        err = nux_session_run(mSession, mInput, mOutput);
        if (err != 0) {
            std::cout << "nux session run error: " << err << std::endl;
        }

#ifdef PROFILE
        nux_profiler_record_begin(mNux, build_profile_record(PROFILE_RECORD_POST_PROCESS, std::to_string(sample.id)).c_str());
#endif
        auto result = nux_tensor_array_get(mOutput, 0);
        nux_buffer_t buffer;
        nux_buffer_len_t len;
        tensor_get_buffer(result, &buffer, &len);
        mResponses[0].id = sample.id;
        mResponses[0].size = len;
#ifdef POST_PROCESS
        post_inference(buffer, mResponses[0]);
#else
        mResponses[0].data = (uintptr_t)&buffer;
#endif

#ifdef PROFILE
        nux_profiler_record_end(mNux, build_profile_record(PROFILE_RECORD_POST_PROCESS, std::to_string(sample.id)).c_str());
#endif
        mlperf::QuerySamplesComplete(mResponses.data(), 1);
#ifdef PROFILE
        nux_profiler_record_end(mNux, build_profile_record(PROFILE_RECORD_INFERENCE, std::to_string(sample.id)).c_str());
#endif
    }
private:
  std::string mName{"FuriosaAI-BlockingSession-SUT"};
};


class FuriosaSUTAlt : public mlperf::SystemUnderTest {
public:
#ifdef ASYNC
    std::unique_ptr<StreamSession> mSession;
#else
    std::unique_ptr<BlockingSession> mSession;
#endif

    FuriosaSUTAlt(int numWorkers = 12) {
#ifdef DEBUG
        std::cout << "Constructing FuriosaSUTAlt" << std::endl;
#endif
#ifdef ASYNC
        mSession.reset(new StreamSession(
			numWorkers, 8, "npu0pe0-1", "mlcommons_resnet50_v1.5_int8.enf"));
        
#else
        mSession.reset(new BlockingSession());
#endif
    }

    ~FuriosaSUTAlt() {
        mSession.reset();
    }

    const std::string &Name() const override { return mSession->Name(); }

    void IssueQuery(const std::vector<mlperf::QuerySample> &samples) override {
#ifdef ASYNC
		mSession->SubmitQuery(samples, 0, 1);
#else
		mSession->Run(samples[0]);
#endif
    }

    void FlushQueries() override {}

    void ReportLatencyResults(
            const std::vector<mlperf::QuerySampleLatency> &latencies_ns) override {};
};


class FuriosaMultiThreadSUT : public mlperf::SystemUnderTest {
public:
    std::vector<std::unique_ptr<StreamSession>> mSessions;

    FuriosaMultiThreadSUT(int numWorkers = 4) {
#ifdef DEBUG
        std::cout << "Constructing FuriosaMultiThreadSUT" << std::endl;
#endif
        for (int peId = 0 ; peId < 2 ; peId++) {
            std::stringstream ostr;
            ostr << "npu0pe" << peId;
            mSessions.emplace_back(new StreamSession(
				numWorkers, 8, ostr.str(), "mlcommons_resnet50_v1.5_int8_batch8.enf"));
        }
    }

    ~FuriosaMultiThreadSUT() {
        mSessions.clear();
    }

    const std::string &Name() const override { return mName; }

    void IssueQuery(const std::vector<mlperf::QuerySample> &samples) override {

#pragma omp parallel for num_threads(2)
        for (int i = 0 ; i < 2 ; i++) {
            auto& session = mSessions[i];
            int half = samples.size() / 2;
            int begin = i * half;
            int end = begin + half;
            int num_iter = (end - begin) / 8;
            int remainder = (end - begin) % 8;
#ifdef DEBUG
            std::cout << "half: " << std::to_string(half) 
                    << ", begin: " << std::to_string(begin)
                    << ", end: " << std::to_string(end)
                    << ", num_iter: " << std::to_string(num_iter)
                    << ", remainder: " << std::to_string(remainder)
					<< std::endl;
#endif
            for (int j = 0 ; j < num_iter ; j++) {
                session->SubmitQuery(samples, begin + j * 8, 8);
            }
            if (remainder > 0) {
                session->SubmitQuery(samples, begin + num_iter * 8, remainder);
            }
        }
    }

    void FlushQueries() override {}

    void ReportLatencyResults(
            const std::vector<mlperf::QuerySampleLatency> &latencies_ns) override {};

private:
  std::string mName{"FuriosaAI-FuriosaMultiThreadSUT"};
};



int main(int argc, char** argv) {
    enable_logging();
    if (getenv("DATA")) {
        DATA_PATH = getenv("DATA");
    }
    if (getenv("VAL_MAP")) {
        VAL_MAP_PATH = getenv("VAL_MAP");
    }

  //assert(argc >= 2 && "Need to pass in at least one argument: target_qps");
  int target_qps = 100;
  bool useQueue{false};
  int numCompleteThreads{4};
  int maxSize{1};
  bool server_coalesce_queries{false};
  int num_issue_threads{0};

  bool isAccuracyRun = false;
  int num_samples = 50000;
  if (argc >= 2 && argv[1][0] >= '0' && argv[1][0] <= '9') {
    num_samples = std::stoi(argv[1]);
    isAccuracyRun = true;
  }

  mlperf::TestSettings testSettings;
  decltype(testSettings.scenario) arg_scenario = mlperf::TestScenario::SingleStream;
  for(int i = 1; i < argc; i ++) {
      if (i + 1 < argc && argv[i] == "--count"s) {
	      num_samples = std::stoi(argv[i+1]);
      }
      if (i + 1 < argc && argv[i] == "--scenario"s) {
          if (argv[i+1] == "SingleStream"s) {
		  std::cout << "Running the SingleStream scenario." << '\n';
          } else if (argv[i+1] == "Offline"s) {
              arg_scenario = mlperf::TestScenario::Offline;
		  std::cout << "Running the Offline scenario." << '\n';
          } else {
              std::cout << "Not supported scenario: " << argv[i+1] << '\n';
              return -1;
          }

      }
      if (argv[i] == "--accuracy"s) {
          isAccuracyRun = true;
      }
  }
  if (num_samples > 50000)
	  num_samples = 50000;
  if (isAccuracyRun) {
	  std::cout << "Accuracy mode." << '\n';
  } else {
	  std::cout << "Performance mode." << '\n';
  }

  QSL qsl(num_samples);
  std::unique_ptr<mlperf::SystemUnderTest> sut;

  // Configure the test settings
  testSettings.FromConfig("../common/mlperf.conf", "resnet50", arg_scenario == mlperf::TestScenario::SingleStream ? "SingleStream" : "Offline");
  testSettings.FromConfig("../common/user.conf", "resnet50", arg_scenario == mlperf::TestScenario::SingleStream ? "SingleStream" : "Offline");
  testSettings.scenario = arg_scenario;
  testSettings.mode = mlperf::TestMode::PerformanceOnly;
  if (isAccuracyRun)
      testSettings.mode = mlperf::TestMode::AccuracyOnly;

  //testSettings.server_coalesce_queries = server_coalesce_queries;
  //std::cout << "testSettings.server_coalesce_queries = "
            //<< (server_coalesce_queries ? "True" : "False") << std::endl;
  //testSettings.server_num_issue_query_threads = num_issue_threads;
  //std::cout << "num_issue_threads = " << num_issue_threads << std::endl;

  // Configure the logging settings
  mlperf::LogSettings logSettings;
  logSettings.log_output.outdir = "build";
  logSettings.log_output.prefix = "mlperf_log_";
  logSettings.log_output.suffix = "";
  logSettings.log_output.prefix_with_datetime = false;
  logSettings.log_output.copy_detail_to_stdout = false;
  logSettings.log_output.copy_summary_to_stdout = true;
  logSettings.log_mode = mlperf::LoggingMode::AsyncPoll;
  logSettings.log_mode_async_poll_interval_ms = 1000;
  logSettings.enable_trace = false;

  // Choose SUT
  if (arg_scenario == mlperf::TestScenario::SingleStream) {
      // sut.reset(new FuriosaBasicSUT());
      sut.reset(new FuriosaSUTAlt(1));
  } else {
      // sut.reset(new FuriosaQueueSUT(2, 8));
      sut.reset(new FuriosaMultiThreadSUT(4));
  }

  // Start test
  std::cout << "Start test..." << std::endl;
  mlperf::StartTest(sut.get(), &qsl, testSettings, logSettings);
  std::cout << "Test done. Clean up SUT..." << std::endl;
  sut.reset();
  std::cout << "Done!" << std::endl;
  return 0;
}
