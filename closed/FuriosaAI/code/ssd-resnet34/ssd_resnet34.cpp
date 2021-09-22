#include <cassert>
#include <cstdio>
#include <chrono>
#include <algorithm>
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
#include <utility>
#include <tuple>
#include "../common/json.hpp"

#include <math.h>
#include <string.h>
#include <stdlib.h>

#include "loadgen.h"
#include "query_sample_library.h"
#include "system_under_test.h"
#include "test_settings.h"

#include "../common/nux.h"

#include <sys/types.h>
#include <dirent.h>
#include "../common/unlower.h"

const int IMAGE_SIZE  = (1200*1200*3);
int INPUT_SIZE = IMAGE_SIZE;
std::optional<LoweringInfo> input_lowering;
std::optional<LoweringInfo> output_lowering[12];

//#define ASYNC

//#define DEBUG

using namespace std::string_literals;

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

std::tuple<
 HCHCW<  8 ,    16 ,     7 ,    21 ,    64, 324, 50, 50>,
 HCHCW<  2 ,    64 ,    13 ,     8 ,    32, 486, 25, 25>,
 HCHWC<  8 ,    16 ,     2 ,    13 ,    32, 486, 13, 13>,
 HCHCW<  1 ,   128 ,     7 ,     4 ,    32, 486, 7, 7>,
 HCHCW<  1 ,   128 ,     3 ,     3 ,    32, 324, 3, 3>,
 HCHCW<  1 ,   128 ,     3 ,     3 ,    32, 324, 3, 3>,
 HCHCW<  8 ,    16 ,     7 ,     1 ,    64, 16, 50, 50>,
 HCHCW<  4 ,    32 ,     7 ,     1 ,    32, 24, 25, 25>,
 HCHCW<  8 ,    16 ,     2 ,     2 ,    32, 24, 13, 13>,
 HCHCW<  4 ,    32 ,     2 ,     1 ,    32, 24, 7, 7>,
 HCHCW<  8 ,    16 ,     1 ,     1 ,    32, 16, 3, 3>,
 HCHCW<  8 ,    16 ,     1 ,     1 ,    32, 16, 3, 3>
> unlower_infos;

const int feature_map_shapes[6] = {50, 25, 13, 7, 3, 3};
const int num_anchors[6] = {4, 6, 6, 6, 4, 4};

std::string DATA_PATH = "../preprocessed/coco-1200-golden/raw/";
std::string META_PATH = "../common/annotations/instances_val2017.json";

std::vector<std::unique_ptr<unsigned char[]>> images;

class Timing {
    std::vector<const char*> names;
    std::vector<int64_t> collected;
    decltype(std::chrono::high_resolution_clock::now()) start_time;
    int count{};
    int current_checkpoint_index{};

public:
    Timing() {
        names.reserve(100);
        collected.reserve(100);
        collected.push_back(0);
    }

    void start() {
        start_time = std::chrono::high_resolution_clock::now();
        current_checkpoint_index = 0;
    }

    void checkpoint(const char* name) {
        auto t = std::chrono::high_resolution_clock::now();
        auto dt = std::chrono::duration_cast<std::chrono::nanoseconds>(t - start_time).count();
        collected[current_checkpoint_index] += dt;
        current_checkpoint_index ++;
        if (names.size() < current_checkpoint_index) {
            names.push_back(name);
            collected.push_back(0);
        }
    }

    void finish() {
        checkpoint("*end*");
        count ++;
    }

    void report() {
        std::cout << "*start*" << '\t';
        for(int i = 0; i < names.size(); i ++) {
            std::cout << (collected[i]-((i>0)?collected[i-1]:0)) / count / 1000000. << '\n' << names[i] << '\t';
        }
        std::cout << '\n';
    }

    ~Timing() {
        report();
    }
} timing;

class QSL : public mlperf::QuerySampleLibrary {
 public:
  QSL(int sampleSize = 5000) 
    : mSampleSize(sampleSize) {
        std::string input_filename;
        int answer;

        std::ifstream meta_file(META_PATH);
        nlohmann::json j;
        meta_file >> j;
        mItems.reserve(5000);

        for(auto& item:j["images"]) {
            std::string fname = item["file_name"].get<std::string>();
            mItems.emplace_back(fname.substr(0,12) + ".jpg.raw", 0);
        }
        images.resize(mItems.size());
    };
  ~QSL() override{};
  const std::string& Name() const override { return mName; }
  size_t TotalSampleCount() override { return mSampleSize; }
  size_t PerformanceSampleCount() override { 
      std::cout << "PerformanceSampleCount" << 1000 << '\n';
      return 1000;
  }
  void LoadSamplesToRam(
          const std::vector<mlperf::QuerySampleIndex>& samples) override {
      for(auto index : samples) {
          //std::string filename = DATA_PATH + mItems[index].first;
          int file_index = index;
          std::string filename = DATA_PATH + mItems[file_index].first;
          //std::cout << "LOAD " << index << filename << '\n';
          std::ifstream inf(filename.c_str(), std::ios::binary);

          if (input_lowering) {
              std::vector<char> buffer(IMAGE_SIZE);
              inf.read((char*)&buffer[0], IMAGE_SIZE);
              std::visit([&](auto info){ 
                  INPUT_SIZE = info.lowered_size();
                  images[index].reset((unsigned char*)std::aligned_alloc(64, info.lowered_size()));
                  info.for_each([&](int c, int co, int ci, int h, int ho, int hi, int w) {
                      images[index][info.index(co, ci, ho, hi, w)] = buffer[c*1200*1200+h*1200+w];
                  });
              }, *input_lowering);
          } else {
              images[index].reset((unsigned char*)std::aligned_alloc(64, IMAGE_SIZE));
              inf.read((char*)&images[index][0], 1200*1200*3);
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

class FuriosaBasicSUT : public mlperf::SystemUnderTest {
 public:
#ifdef ASYNC
    nux_async_session_t mSession;
#else
    nux_session_t mSession;
#endif
    nux_completion_queue_t mQueue;
    nux_session_option_t mSessionOption;
    nux_tensor_array_t mInputs;
    nux_tensor_array_t mOutputs;
    std::atomic<int> mIssued;
    std::atomic<int> mCompleted;

    const int num_classes = 81;

    const float scale_xy = 0.1f;
    const float scale_wh = 0.2f;

  FuriosaBasicSUT() 
  {
      scores3.resize(81);
    // Start with some large value so that we don't reallocate memory.
    initResponse(1);

    //std::ifstream inf("mlcommons_ssd_resnet34_int8.enf", std::ios::binary);
    std::ifstream inf("mlcommons_ssd_resnet34_int8.enf", std::ios::binary);

    mSessionOption = nux_session_option_create();
    nux_session_option_set_device(mSessionOption, "npu0pe0-1");

    std::vector<char> model_buf((std::istreambuf_iterator<char>(inf)),
                     std::istreambuf_iterator<char>());

    {
#ifdef ASYNC
        auto err = nux_async_session_create((nux_buffer_t)model_buf.data(), model_buf.size(), mSessionOption, &mSession, &mQueue);
#else
        auto err = nux_session_create((nux_buffer_t)model_buf.data(), model_buf.size(), mSessionOption, &mSession);
#endif
        if (err != nux_error_t_success) {
            std::cerr << "SUT:nux async session create error: " << err << '\n';
            exit(-1);
        }
    }

#ifdef ASYNC
    nux_model_t model = nux_async_session_get_model(mSession);
#else
    nux_model_t model = nux_session_get_model(mSession);
#endif
    mInputs = nux_tensor_array_create_inputs(model);
    mOutputs = nux_tensor_array_allocate_outputs(model);

    auto input_desc = nux_input_desc(model, 0);
    input_lowering = TensorDescToLoweringInfo(input_desc, 3, 1200, 1200);
    if (input_lowering) {
        std::cout << "Using pre-lowered input.\n";
    }

    auto output_desc =  nux_output_desc(model, 0);
    output_lowering[0] = TensorDescToLoweringInfo(output_desc, num_classes*4, 50, 50);
    output_desc =  nux_output_desc(model, 1);
    output_lowering[1] = TensorDescToLoweringInfo(output_desc, num_classes*6, 25, 25);
    output_desc =  nux_output_desc(model, 2);
    output_lowering[2] = TensorDescToLoweringInfo(output_desc, num_classes*6, 13, 13);
    output_desc =  nux_output_desc(model, 3);
    output_lowering[3] = TensorDescToLoweringInfo(output_desc, num_classes*6, 7, 7);
    output_desc =  nux_output_desc(model, 4);
    output_lowering[4] = TensorDescToLoweringInfo(output_desc, num_classes*4, 3, 3);
    output_desc =  nux_output_desc(model, 5);
    output_lowering[5] = TensorDescToLoweringInfo(output_desc, num_classes*4, 3, 3);
    output_desc =  nux_output_desc(model, 6);
    output_lowering[6] = TensorDescToLoweringInfo(output_desc, 4*4, 50, 50);
    output_desc =  nux_output_desc(model, 7);
    output_lowering[7] = TensorDescToLoweringInfo(output_desc, 4*6, 25, 25);
    output_desc =  nux_output_desc(model, 8);
    output_lowering[8] = TensorDescToLoweringInfo(output_desc, 4*6, 13, 13);
    output_desc =  nux_output_desc(model, 9);
    output_lowering[9] = TensorDescToLoweringInfo(output_desc, 4*6, 7, 7);
    output_desc =  nux_output_desc(model, 10);
    output_lowering[10] = TensorDescToLoweringInfo(output_desc, 4*4, 3, 3);
    output_desc =  nux_output_desc(model, 11);
    output_lowering[11] = TensorDescToLoweringInfo(output_desc, 4*4, 3, 3);

#ifdef ASYNC
    mCompleteThreads.push_back(std::thread([this]{CompleteThread();}));
#endif

    for(int i = 0; i < 12; i ++) {
        float s = output_dequantization_params[i][0];
        float z = output_dequantization_params[i][1];
        output_dequantization_tables[i].resize(256);
        output_exp_dequantization_tables[i].resize(256);
        output_exp_scale_dequantization_tables[i].resize(256);
        for(int q = -128; q < 128; q++) {
            float x = s*(q-z);
            output_dequantization_tables[i][(q<0)?q+256:q] = x;
            output_exp_dequantization_tables[i][(q<0)?q+256:q] = expf(x);
            output_exp_scale_dequantization_tables[i][(q<0)?q+256:q] = expf(x*scale_wh);

        }
    }

    {
        std::ifstream prior_file("ssd_large_precomputed_priors", std::ios::binary);
        prior_file.read((char*)priors, 15130*4*4);
    }
  }


std::array<float, 15130*4> decode_box(
        unsigned char* (&buffers)[12],
        nux_buffer_len_t (&lens)[12]
) {
    std::array<float, 15130*4> boxes3;
    int output_base_index[12] = {
        0,
        10000, 
        10000+3750,
        10000+3750+1014,
        10000+3750+1014+294,
        10000+3750+1014+294+36,
        0,
        10000, 
        10000+3750,
        10000+3750+1014,
        10000+3750+1014+294,
        10000+3750+1014+294+36,
        //36,
    };

	for_each(unlower_infos, [&](int output_index, auto info) {
        if (output_index < 6)
            return;
            {
        auto& buffers_lowered = buffers[output_index];

        //box
        auto tile_index = output_index-6;
        //C(point * anchor) H W
        //anchor H W point

        const auto num_anchor = num_anchors[tile_index];
        const auto f_w = feature_map_shapes[tile_index];
        const auto f_h = feature_map_shapes[tile_index];
        const int num_point = 4;

        const int point_stride_for_boxes = 1;
        const int w_stride_for_boxes = num_point * point_stride_for_boxes;
        const int h_stride_for_boxes = f_w * w_stride_for_boxes;
        const int anchor_stride_for_boxes = f_h * h_stride_for_boxes;

        for (int h_index = 0; h_index < f_h; ++ h_index) {
            for (int c_index = 0; c_index < num_anchor; ++c_index) {
                for (int w_index = 0; w_index < f_w; ++ w_index) {

                    const auto q0 = (unsigned char)buffers_lowered[info.index(c_index+0*num_anchor, h_index, w_index)];
                    const auto q1 = (unsigned char)buffers_lowered[info.index(c_index+1*num_anchor, h_index, w_index)];
                    const auto q2 = (unsigned char)buffers_lowered[info.index(c_index+2*num_anchor, h_index, w_index)];
                    const auto q3 = (unsigned char)buffers_lowered[info.index(c_index+3*num_anchor, h_index, w_index)];

                    const auto d_q0 = (output_dequantization_tables[output_index][q0]);
                    const auto d_q1 = (output_dequantization_tables[output_index][q1]);
                    const auto d_q2 = (output_exp_scale_dequantization_tables[output_index][q2]);
                    const auto d_q3 = (output_exp_scale_dequantization_tables[output_index][q3]);

                    const int n = output_base_index[output_index] + c_index*f_w*f_h + h_index*f_w + w_index;
                    const auto x = d_q0 * scale_xy * priors[n][2] + priors[n][0];
                    const auto y = d_q1 * scale_xy * priors[n][3] + priors[n][1];
                    const auto w = d_q2*priors[n][2];
                    const auto h = d_q3*priors[n][3];


                    const auto anchor_index = c_index;

                    auto b0_index = 4*output_base_index[output_index]
                        + w_index * w_stride_for_boxes
                        + h_index * h_stride_for_boxes
                        + anchor_index * anchor_stride_for_boxes
                        + 0 * point_stride_for_boxes;

                    auto b1_index = 4*output_base_index[output_index]
                        + w_index * w_stride_for_boxes
                        + h_index * h_stride_for_boxes
                        + anchor_index * anchor_stride_for_boxes
                        + 1 * point_stride_for_boxes;

                    auto b2_index = 4*output_base_index[output_index]
                        + w_index * w_stride_for_boxes
                        + h_index * h_stride_for_boxes
                        + anchor_index * anchor_stride_for_boxes
                        + 2 * point_stride_for_boxes;

                    auto b3_index = 4*output_base_index[output_index]
                        + w_index * w_stride_for_boxes
                        + h_index * h_stride_for_boxes
                        + anchor_index * anchor_stride_for_boxes
                        + 3 * point_stride_for_boxes;

                    boxes3[b0_index] = x - 0.5f*w;
                    boxes3[b1_index] = y - 0.5f*h;
                    boxes3[b2_index] = x + 0.5f*w;
                    boxes3[b3_index] = y + 0.5f*h;

                }
            }
        }
    }
    });
    return boxes3;
}

  //Ort::Session mPostprocessOrtSession{env, "mlcommons_ssd_resnet34_postprocess.onnx", Ort::SessionOptions{nullptr}};
  float priors[15130][4];

  const float output_dequantization_params[12][2] = {
//ssd1200/multibox_head/cls_0/BiasAdd:0_quantized {'s': 0.13746771216392517, 'z': -34}
//ssd1200/multibox_head/cls_1/BiasAdd:0_quantized {'s': 0.1921287328004837, 'z': -53}
//ssd1200/multibox_head/cls_2/BiasAdd:0_quantized {'s': 0.12818175554275513, 'z': -33}
//ssd1200/multibox_head/cls_3/BiasAdd:0_quantized {'s': 0.12459244579076767, 'z': -31}
//ssd1200/multibox_head/cls_4/BiasAdd:0_quantized {'s': 0.11598147451877594, 'z': -34}
//ssd1200/multibox_head/cls_5/BiasAdd:0_quantized {'s': 0.10715563595294952, 'z': -54}
//ssd1200/multibox_head/loc_0/BiasAdd:0_quantized {'s': 0.07815917581319809, 'z': 53}
//ssd1200/multibox_head/loc_1/BiasAdd:0_quantized {'s': 0.03388494625687599, 'z': 6}
//ssd1200/multibox_head/loc_2/BiasAdd:0_quantized {'s': 0.032343193888664246, 'z': 0}
//ssd1200/multibox_head/loc_3/BiasAdd:0_quantized {'s': 0.029285095632076263, 'z': -9}
//ssd1200/multibox_head/loc_4/BiasAdd:0_quantized {'s': 0.0338519886136055, 'z': 0}
//ssd1200/multibox_head/loc_5/BiasAdd:0_quantized {'s': 0.023822087794542313, 'z': 3}

{0.13746771216392517, -34},
{0.1921287328004837, -53},
{0.12818175554275513, -33},
{0.12459244579076767, -31},
{0.11598147451877594, -34},
{0.10715563595294952, -54},
{0.07815917581319809, 53},
{0.03388494625687599, 6},
{0.032343193888664246, 0},
{0.029285095632076263, -9},
{0.0338519886136055, 0},
{0.023822087794542313, 3},
  };
  std::vector<float> output_dequantization_tables[12];
  std::vector<float> output_exp_dequantization_tables[12];
  std::vector<float> output_exp_scale_dequantization_tables[12];

  std::vector<std::thread> mCompleteThreads;
  std::vector<float> results;

  long long time_used_on_postprocess = 0;
  int postproces_count = 0;

  ~FuriosaBasicSUT() override {
#ifdef ASYNC
      nux_completion_queue_destroy(mQueue);
      nux_async_session_destroy(mSession);
#else
      nux_session_destroy(mSession);
#endif
      if (postproces_count > 0) {
          std::cout << "POSTPROCESS TIME (avg): " << time_used_on_postprocess*1.0/postproces_count << '\n';
      }
  }

std::vector<std::array<float, 15130>> scores3;
  const std::string& Name() const override { return mName; }
  void IssueQuery(const std::vector<mlperf::QuerySample>& samples) override {
      results.clear();
      results.reserve(200*7);
    int n = samples.size();
    for (int i = 0; i < n; i++) {
        mResponses[0].id = samples[i].id;
        auto tensor0 = nux_tensor_array_get(mInputs, 0);
        auto* data = images[samples[i].index].get();

        //std::cout << "Issuing: " << samples[i].index << ' ' << i << '/' << n << ' ' << '\n';
        //tensor_set_buffer(tensor0, (nux_buffer_t)dummy.data(), dummy.size(), nullptr);
        auto err = tensor_set_buffer(tensor0, (nux_buffer_t)data, INPUT_SIZE, nullptr);

#ifdef ASYNC
        nux_async_session_run(mSession, (nux_context_t)samples[i].id, mInputs);
#else
        err = nux_session_run(mSession, mInputs, mOutputs);
        if (err != 0) {
            std::cout << "Error: " << err << '\n';
        }

        timing.start();

        nux_buffer_t buffers[12];
        nux_buffer_len_t lens[12];
        thread_local std::vector<float> outputs[2];
        outputs[0].resize(15130*81);
        outputs[1].resize(15130*4);


        timing.checkpoint("read buffers");
        for(int output_index = 0; output_index < 12; output_index ++) {
            auto result = nux_tensor_array_get(mOutputs, output_index);
            err = tensor_get_buffer(result, &buffers[output_index], &lens[output_index]);
        }
        timing.checkpoint("decode box");
        auto boxes3 = decode_box(buffers, lens);

        timing.checkpoint("decode score");

        const int output_base_index[12] = {
            0,
            10000, 
            10000+3750,
            10000+3750+1014,
            10000+3750+1014+294,
            10000+3750+1014+294+36,
            0,
            10000, 
            10000+3750,
            10000+3750+1014,
            10000+3750+1014+294,
            10000+3750+1014+294+36,
            //36,
        };


        // 81x15130 -> 15130x81 / 4x15130 -> 15130x4
        //std::array<float, 81*15130> scores3;
        // avoid false sharing (using only first element of inner array)
        std::array<std::array<float,16>, 15130> scores_sum3;

#pragma omp parallel for num_threads(12)
        for(int class_index = 0; class_index < num_classes; class_index ++)
        for_each(unlower_infos, [&](int output_index, auto info){
            if (output_index >= 6)
                return;
            {
                auto tile_index = output_index;
                auto num_anchor = num_anchors[tile_index];

                    for(int anchor_index = 0; anchor_index < num_anchor; anchor_index++)
                        for(int h = 0; h < info.NHu; h++)
                        for(int w = 0; w < info.NWu; w++)
                        {
                            const auto q = buffers[output_index][info.index(class_index * num_anchor + anchor_index,h,w)];
                            const auto score = output_exp_dequantization_tables[output_index][q];
                            const int scores_sum_index = anchor_index * info.NHu * info.NWu + h * info.NWu + w  + output_base_index[output_index];
                            scores3[class_index][scores_sum_index] = score;
                        }
            }
        });
#pragma omp parallel for num_threads(12)
        for(int i = 0; i < 15130; i ++) { 
            scores_sum3[i][0] = 0;
            for(int class_index = 0; class_index < num_classes; class_index ++)
                scores_sum3[i][0] += scores3[class_index][i];
        }

        const auto score_threshold = 0.05f;
        const auto nms_threshold = 0.5f;


        timing.checkpoint("NMS");
        auto picked = filter_results(scores3, boxes3, scores_sum3, score_threshold, nms_threshold);
#ifdef DEBUG
        std::cout << "picked: " << picked.size() << '\n';
#endif

        timing.checkpoint("build data for loadgen");
        results.clear();
        for(const auto& [score, class_index_and_box] :picked) {
            auto box = class_index_and_box % 15130;
            auto class_index = class_index_and_box / 15130;
            results.push_back(samples[i].index);
            results.push_back(boxes3[box*4+1]);
            results.push_back(boxes3[box*4+0]);
            results.push_back(boxes3[box*4+3]);
            results.push_back(boxes3[box*4+2]);
            results.push_back(score);
            results.push_back(class_index);
#ifdef DEBUG
            std::cout << "    " << i << ' ' << samples[i].index << ' ' << results[results.size()-2] << ' ' << results[results.size()-1] << '\n';
            std::cout << "        " << boxes3[box*4+1]<< ' ' << boxes3[box*4+0]<< ' ' << boxes3[box*4+3]<< ' ' << boxes3[box*4+2]<< '\n';
#endif
        }

        timing.finish();

        mResponses[0].data = (uintptr_t)&results[0];
        mResponses[0].size = results.size() * sizeof(float);
        mlperf::QuerySamplesComplete(mResponses.data(), 1);
        //nux_tensor_array_destroy(outputs);
#endif
    }

  }

  void FlushQueries() override {}
  void ReportLatencyResults(
      const std::vector<mlperf::QuerySampleLatency>& latencies_ns) override{};

  std::vector<std::tuple<float, int>> filter_results(std::vector<std::array<float, 15130>>& scores, std::array<float, 15130*4>& boxes, std::array<std::array<float,16>, 15130>& scores_sum, float score_threshold = 0.3f, float nms_threshold = 0.6f) {

      const auto iou = [](float* b1, float* b2)->float {
        const float eps = 1e-5f;
        const auto clamp_lower = [](float x)->float{ if (x < 0) return 0; return x; };
        float area1 = clamp_lower(b1[3]-b1[1]) * clamp_lower(b1[2]-b1[0]);
        float area2 = clamp_lower(b2[3]-b2[1]) * clamp_lower(b2[2]-b2[0]);
        float cw = clamp_lower(std::min(b1[3], b2[3])-std::max(b1[1],b2[1]));
        float ch = clamp_lower(std::min(b1[2], b2[2])-std::max(b1[0],b2[0]));
        float overlap = cw*ch;
        return overlap / (area1+area2-overlap+eps);
      };

      std::vector<std::tuple<float, int>> result;
      result.reserve(num_classes * 200);
#pragma omp parallel for num_threads(12)
      for(int class_index = 1; class_index < num_classes; class_index++) {
		  thread_local std::vector<std::tuple<float, int>> filtered2;
		  filtered2.reserve(15130);
		  filtered2.clear();
          // filtering low scores before nms
          for(int i = 0; i < 15130; i ++) {
              auto score = scores[class_index][i];
              if (score / scores_sum[i][0] > score_threshold) {
                  filtered2.emplace_back(score/scores_sum[i][0], i);
              }
          }

          // look through boxes in the descending order
          auto part_end2 = filtered2.size() > 200 ? filtered2.begin()+200 : filtered2.end();
          std::partial_sort(filtered2.begin(), part_end2, filtered2.end(), [](auto l, auto r) {
              return std::get<0>(l) > std::get<0>(r);
          });
          filtered2.erase(part_end2, filtered2.end());


          std::vector<std::tuple<float, int>> l_picked;
          l_picked.reserve(200);

		  int cnt2 = 0;
          for(auto [l_score, l_boxes_index] :filtered2) {
			 if (cnt2 < 200) {
                bool can_pick = true;
                for(auto p = 0; p < l_picked.size(); ++p) {
                  auto [r_score, r_boxes_index] = l_picked[p];
                  if (iou(&boxes[l_boxes_index*4], &boxes[(r_boxes_index%15130)*4]) > nms_threshold) {
                      can_pick = false;
                      break;
                  }
                }
                if (can_pick) {
                  cnt2++;
                  //l_picked.emplace_back(class_index, l_score, l_boxes_index);
                  l_picked.emplace_back(l_score, l_boxes_index+class_index*15130);
                }
			 }

          }
#pragma omp critical
          result.insert(result.end(), l_picked.begin(), l_picked.end());
      }
	  if (result.size() > 200) {
          std::partial_sort(result.begin(), result.begin()+200, result.end(), [&](const auto& l, const auto& r) {
              return std::get<0>(l) > std::get<0>(r);
          });
          result.erase(result.begin()+200, result.end());
	  }
	  return result;
  }

#ifdef ASYNC
  void CompleteThread() {
      std::cout << "Begin Completion Thread" << '\n';
      nux_context_t context;
      nux_tensor_array_t outputs;
      enum nux_error_t error;
      while(nux_completion_queue_next(mQueue,
          &context,
          &outputs,
          &error)) {

        intptr_t response_id = (intptr_t)context;
        std::cout << "Completing: " << response_id << '\n';

        auto tensor1 = nux_tensor_array_get(outputs, 0);
        auto tensor2 = nux_tensor_array_get(outputs, 1);
        mlperf::QuerySamplesComplete(mResponses.data(), 1);

      }
    
  }
#endif

 private:
  void initResponse(int size) {
    mResponses.resize(size,
                      {0, reinterpret_cast<uintptr_t>(&mBuf), sizeof(int)});
  }
  int mBuf{0};
  std::string mName{"FuriosaAI-BasicSUT"};
  std::vector<mlperf::QuerySampleResponse> mResponses;
};

class QueueSUT : public mlperf::SystemUnderTest {
 public:
  QueueSUT(int numCompleteThreads, int maxSize) {
    // Each thread handle at most maxSize at a time.
    std::cout << "QueueSUT: maxSize = " << maxSize << std::endl;
    initResponse(numCompleteThreads, maxSize);
    // Launch complete threads
    for (int i = 0; i < numCompleteThreads; i++) {
      mThreads.emplace_back(&QueueSUT::CompleteThread, this, i);
    }
  }
  ~QueueSUT() override {
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
    std::unique_lock<std::mutex> lck(mMtx);
    for (const auto& sample : samples) {
      mIdQueue.push_back(sample.id);
    }
    // Let some worker thread to consume tasks
    mCondVar.notify_one();
  }
  void FlushQueries() override {}
  void ReportLatencyResults(
      const std::vector<mlperf::QuerySampleLatency>& latencies_ns) override{};

 private:
  void CompleteThread(int threadIdx) {
    auto& responses = mResponses[threadIdx];
    size_t maxSize{responses.size()};
    size_t actualSize{0};
    while (true) {
      {
        std::unique_lock<std::mutex> lck(mMtx);
        mCondVar.wait(lck, [&]() { return !mIdQueue.empty() || mDone; });

        if (mDone) {
          break;
        }

        actualSize = std::min(maxSize, mIdQueue.size());
        for (int i = 0; i < actualSize; i++) {
          responses[i].id = mIdQueue.front();
          mIdQueue.pop_front();
        }
        mCondVar.notify_one();
      }
      mlperf::QuerySamplesComplete(responses.data(), actualSize);
    }
  }
  void initResponse(int numCompleteThreads, int size) {
    mResponses.resize(numCompleteThreads);
    for (auto& responses : mResponses) {
      responses.resize(size,
                       {0, reinterpret_cast<uintptr_t>(&mBuf), sizeof(int)});
    }
  }
  int mBuf{0};
  std::string mName{"QueueSUT"};
  std::vector<std::vector<mlperf::QuerySampleResponse>> mResponses;
  std::vector<std::thread> mThreads;
  std::deque<mlperf::ResponseId> mIdQueue;
  std::mutex mMtx;
  std::condition_variable mCondVar;
  bool mDone{false};
};

class MultiBasicSUT : public mlperf::SystemUnderTest {
 public:
  MultiBasicSUT(int numThreads)
      : mNumThreads(numThreads), mResponses(numThreads) {
    // Start with some large value so that we don't reallocate memory.
    initResponse(10000);
    for (int i = 0; i < mNumThreads; ++i) {
      mThreads.emplace_back(&MultiBasicSUT::startIssueThread, this, i);
    }
  }
  ~MultiBasicSUT() override {
    for (auto& thread : mThreads) {
      thread.join();
    }
  }
  const std::string& Name() const override { return mName; }
  void IssueQuery(const std::vector<mlperf::QuerySample>& samples) override {
    int thread_idx = mThreadMap[std::this_thread::get_id()];
    int n = samples.size();
    auto& reponses = mResponses[thread_idx];
    if (n > reponses.size()) {
      std::cout
          << "Warning: reallocating response buffer in MultiBasicSUT. Maybe "
             "you should initResponse with larger value!?"
          << std::endl;
      initResponse(samples.size());
    }
    for (int i = 0; i < n; i++) {
      reponses[i].id = samples[i].id;
    }
    mlperf::QuerySamplesComplete(reponses.data(), n);
  }
  void FlushQueries() override {}
  void ReportLatencyResults(
      const std::vector<mlperf::QuerySampleLatency>& latencies_ns) override{};

 private:
  void initResponse(int size) {
    for (auto& responses : mResponses) {
      responses.resize(size,
                       {0, reinterpret_cast<uintptr_t>(&mBuf), sizeof(int)});
    }
  }
  void startIssueThread(int thread_idx) {
    {
      std::lock_guard<std::mutex> lock(mMtx);
      mThreadMap[std::this_thread::get_id()] = thread_idx;
    }
    mlperf::RegisterIssueQueryThread();
  }
  int mBuf{0};
  int mNumThreads{0};
  std::string mName{"MultiBasicSUT"};
  std::vector<std::vector<mlperf::QuerySampleResponse>> mResponses;
  std::mutex mMtx;
  std::vector<std::thread> mThreads;
  std::map<std::thread::id, int> mThreadMap;
};

int main(int argc, char** argv) {
    enable_logging();

    if (getenv("DATA")) {
        DATA_PATH = getenv("DATA");
    }
    if (getenv("META")) {
        META_PATH = getenv("META");
    }

  //assert(argc >= 2 && "Need to pass in at least one argument: target_qps");
  int num_samples = 5000;
  bool isAccuracyRun = false;
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
          } else if (argv[i+1] == "Offline"s) {
              arg_scenario = mlperf::TestScenario::Offline;
          } else {
              std::cout << "Not supported scenario: " << argv[i+1] << '\n';
              return -1;
          }

      }
      if (argv[i] == "--accuracy"s) {
          isAccuracyRun = true;
      }
  }
  if (num_samples > 5000)
	  num_samples = 5000;

  QSL qsl(num_samples);
  std::unique_ptr<mlperf::SystemUnderTest> sut;

  int target_qps = 100;
  bool useQueue{false};
  int numCompleteThreads{4};
  int maxSize{1};
  bool server_coalesce_queries{false};
  int num_issue_threads{0};

  testSettings.FromConfig("../common/mlperf.conf", "ssd-resnet34", arg_scenario == mlperf::TestScenario::SingleStream ? "SingleStream" : "Offline");
  testSettings.FromConfig("../common/user.conf", "ssd-resnet34", arg_scenario == mlperf::TestScenario::SingleStream ? "SingleStream" : "Offline");
  // Configure the test settings
  testSettings.scenario = arg_scenario;
  testSettings.mode = mlperf::TestMode::PerformanceOnly;
  if (isAccuracyRun)
      testSettings.mode = mlperf::TestMode::AccuracyOnly;

  // testSettings.server_coalesce_queries = server_coalesce_queries;
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
  if (num_issue_threads == 0) {
    if (useQueue) {
      std::cout << "Using QueueSUT with " << numCompleteThreads
                << " complete threads" << std::endl;
      sut.reset(new QueueSUT(numCompleteThreads, maxSize));
    } else {
      std::cout << "Using BasicSUT" << std::endl;
      sut.reset(new FuriosaBasicSUT());
    }
  } else {
    if (useQueue) {
      std::cout << "Using MultiQueueSUT with " << numCompleteThreads
                << " complete threads" << std::endl;
      std::cerr << "!!!! MultiQueueSUT is NOT implemented yet !!!!"
                << std::endl;
      return 1;
      // sut.reset(new MultiQueueSUT(num_issue_threads, numCompleteThreads,
      // maxSize));
    } else {
      std::cout << "Using MultiBasicSUT" << std::endl;
      sut.reset(new MultiBasicSUT(num_issue_threads));
    }
  }

  // Start test
  std::cout << "Start test..." << std::endl;
  mlperf::StartTest(sut.get(), &qsl, testSettings, logSettings);
  std::cout << "Test done. Clean up SUT..." << std::endl;
  sut.reset();
  std::cout << "Done!" << std::endl;
  return 0;
}
