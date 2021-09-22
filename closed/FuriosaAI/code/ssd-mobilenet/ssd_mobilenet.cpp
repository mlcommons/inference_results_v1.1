#include <cassert>
#include <sstream>
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
#include "../common/json.hpp"

#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "loadgen.h"
#include "query_sample_library.h"
#include "system_under_test.h"
#include "test_settings.h"

#include "../common/nux.h"

#include <sys/types.h>
#include <dirent.h>

#include "../common/unlower.h"

const int IMAGE_SIZE = (300*300*3);
int INPUT_SIZE = IMAGE_SIZE;
int OUTPUT_SIZE[12];
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

//std::tuple<
//HCHCW< 1,   128,    19,     3,    32, 273, 19, 19>,
//HCHCW< 1,   128,    10,     5,    32, 546, 10, 10>,
//HCHCW< 1,   128,     5,     5,    32, 546, 5, 5>,
//HCHCW< 1,   128,     3,     5,    32, 546, 3, 3>,
//HCHCW< 1,   128,     2,     5,    32, 546, 2, 2>,
//HCHCW< 1,   128,     1,     5,    32, 546, 1, 1>,
//HCHCW< 8,    16,     3,     1,    32, 12, 19, 19>,
//HCHCW< 8,    16,     2,     2,    32, 24, 10, 10>,
//HCHCW< 1,   128,     5,     1,    32, 24, 5, 5>,
//HCHCW< 1,   128,     3,     1,    32, 24, 3, 3>,
//HCHCW< 1,   128,     2,     1,    32, 24, 2, 2>,
//HCHCW< 1,   128,     1,     1,    32, 24, 1, 1>
//> unlower_infos;

std::string DATA_PATH = "../preprocessed/coco-300-golden/raw/";
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
        if (count)
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
            mItems.emplace_back(fname.substr(0,12) + ".png.raw", 0);
        }
        images.resize(mItems.size());
    };
  ~QSL() override{};
  const std::string& Name() const override { return mName; }
  size_t TotalSampleCount() override { return mSampleSize; }
  size_t PerformanceSampleCount() override { 
      const int n = 1024;
      std::cout << "PerformanceSampleCount " << 1024 << '/' << TotalSampleCount() << '\n';
      return 512;
  }
  void LoadSamplesToRam(
          const std::vector<mlperf::QuerySampleIndex>& samples) override {
      for(auto index : samples) {
          std::string filename = DATA_PATH + mItems[index].first;
          //std::cout << "LOAD " << index << filename << '\n';
          std::ifstream inf(filename.c_str(), std::ios::binary);
          if (input_lowering) {
              std::vector<char> buffer(IMAGE_SIZE);
              inf.read((char*)&buffer[0], IMAGE_SIZE);
              std::visit([&](auto info){ 
                  INPUT_SIZE = info.lowered_size();
                  images[index].reset((unsigned char*)std::aligned_alloc(64, info.lowered_size()));
                  info.for_each([&](int c, int co, int ci, int h, int ho, int hi, int w) {
                      images[index][info.index(co, ci, ho, hi, w)] = buffer[c*300*300+h*300+w];
                  });
              }, *input_lowering);
          } else {
              images[index].reset((unsigned char*)std::aligned_alloc(64, IMAGE_SIZE));
              inf.read((char*)&images[index][0], 300*300*3);
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

class SSDMoblienetReporter {
    protected:

        SSDMoblienetReporter() {
            {
                std::ifstream prior_file("ssd_small_precomputed_priors", std::ios::binary);
                prior_file.read((char*)priors, 1917*4*4);
            }

            prior_base_index[0] = 0;
            for(int i = 1; i < 6; i++) {
                prior_base_index[i] = prior_base_index[i-1] + feature_map_shapes[i-1]*feature_map_shapes[i-1]*num_anchors[i-1];
            }

            for(int i = 0; i < 12; i ++) {
                float s = output_dequantization_params[i][0];
                float z = output_dequantization_params[i][1];
                for(int q = -128; q < 128; q++) {
                    float x = s*(q-z);
                    if (i < 6) {
                        // x = sigmoid(x)
                        x = exp(x)/(1+exp(x));
                    }
                    int q_idx = q;
                    if (q_idx < 0) q_idx += 256;
                    output_dequantization_tables[i][q_idx] = x;
                    output_exp_dequantization_tables[i][q_idx] = expf(x/coder_weight[2]);
                }
            }
        }

  const int feature_map_shapes[6] = {19, 10, 5, 3, 2, 1};
  const int num_anchors[6] = {3, 6, 6, 6, 6, 6};
  static const int num_classes = 91;
  const float coder_weight[6] = {10, 10, 5, 5};
  float priors[1917][4];
  int prior_base_index[6];
  const float output_dequantization_params[12][2] = {
     //class_logit_0_quantized {'s': 0.1721179485321045, 'z': 80}
     //class_logit_1_quantized {'s': 0.15295906364917755, 'z': 45}
     //class_logit_2_quantized {'s': 0.11843560636043549, 'z': 75}
     //class_logit_3_quantized {'s': 0.10968340933322906, 'z': 73}
     //class_logit_4_quantized {'s': 0.09295836091041565, 'z': 72}
     //class_logit_5_quantized {'s': 0.10040207952260971, 'z': 64}
     //box_regression_0_quantized {'s': 0.09716527909040451, 'z': 62}
     //box_regression_1_quantized {'s': 0.04244277998805046, 'z': 27}
     //box_regression_2_quantized {'s': 0.03502378985285759, 'z': -5}
     //box_regression_3_quantized {'s': 0.033651404082775116, 'z': -12}
     //box_regression_4_quantized {'s': 0.02295185998082161, 'z': 14}
     //box_regression_5_quantized {'s': 0.023176569491624832, 'z': 27}

     {0.1721179485321045, 80},
     {0.15295906364917755, 45},
     {0.11843560636043549, 75},
     {0.10968340933322906, 73},
     {0.09295836091041565, 72},
     {0.10040207952260971, 64},

     {0.09716527909040451, 62},
     {0.04244277998805046, 27},
     {0.03502378985285759, -5},
     {0.033651404082775116, -12},
     {0.02295185998082161, 14},
     {0.023176569491624832, 27},
  };

    std::array<float, 256> output_dequantization_tables[12];
    std::array<float, 256> output_exp_dequantization_tables[12];

    void post_inference(
                float result_index,
                unsigned char** buffers,
                std::vector<float>& results
            ) {

        //timing.start();
        //timing.checkpoint("decode box");
// prior: [19 * 19 * 3] * [4]
// OUTPUT:
// classes
// 98553 54600 13650 4914 2184 546
// [3(anchor) * 91(num_classes)] * [19] * [19]
// boxes
// 4332 2400 600 216 96 24
// [3(anchor) * 4(xywh)] * [19] * [19]
    std::array<float, 1917*4> boxes;

    const std::array<int, 6> output_index_stride_map_for_boxes = {0, 1083, 1683, 1833, 1887, 1911 };


    int b_counter = 0;
    for(int output_index = 6; output_index < 12; output_index++) 
        std::visit([&](auto info){
        //if (output_index < 6)
            //return;
        int tile_index = output_index - 6;

        const int point_stride_for_boxes = 1;
        const int feature_stride_for_boxes = 4 * point_stride_for_boxes;
        const int anchor_stride_for_boxes = feature_map_shapes[tile_index] * feature_map_shapes[tile_index] * feature_stride_for_boxes;

        //buffer: tile, anchor, shape^2
        //boxes: tile, anchor, shape^2, point
        for(int anchor_index = 0; anchor_index < num_anchors[tile_index]; anchor_index ++) {
            for(int f_y = 0; f_y < feature_map_shapes[tile_index]; f_y++)
            for(int f_x = 0; f_x < feature_map_shapes[tile_index]; f_x++) {
                const int feature_index = f_y * feature_map_shapes[tile_index] + f_x;
                //buffer: tile, anchor, point, shape^2

                //auto q0 = (unsigned char)buffers[output_index][anchor_index * anchor_stride_for_box_buffer + 0 * point_stride_for_box_buffer + feature_index * feature_stride_for_box_buffer];
                //auto q1 = (unsigned char)buffers[output_index][anchor_index * anchor_stride_for_box_buffer + 1 * point_stride_for_box_buffer + feature_index * feature_stride_for_box_buffer];
                //auto q2 = (unsigned char)buffers[output_index][anchor_index * anchor_stride_for_box_buffer + 2 * point_stride_for_box_buffer + feature_index * feature_stride_for_box_buffer];
                //auto q3 = (unsigned char)buffers[output_index][anchor_index * anchor_stride_for_box_buffer + 3 * point_stride_for_box_buffer + feature_index * feature_stride_for_box_buffer];
                auto q0 = (unsigned char)buffers[output_index][info.index(anchor_index*4+0, f_y, f_x)];
                auto q1 = (unsigned char)buffers[output_index][info.index(anchor_index*4+1, f_y, f_x)];
                auto q2 = (unsigned char)buffers[output_index][info.index(anchor_index*4+2, f_y, f_x)];
                auto q3 = (unsigned char)buffers[output_index][info.index(anchor_index*4+3, f_y, f_x)];

                float x = output_dequantization_tables[output_index][q1] / coder_weight[1];
                float y = output_dequantization_tables[output_index][q0] / coder_weight[0];
                //float w = expf(output_dequantization_tables[output_index][q3] / coder_weight[3]);
                //float h = expf(output_dequantization_tables[output_index][q2] / coder_weight[2]);
                float w = output_exp_dequantization_tables[output_index][q3];
                float h = output_exp_dequantization_tables[output_index][q2];

                float px1 = priors[prior_base_index[tile_index] + feature_index*num_anchors[tile_index] + anchor_index][1];
                float py1 = priors[prior_base_index[tile_index] + feature_index*num_anchors[tile_index] + anchor_index][0];
                float px2 = priors[prior_base_index[tile_index] + feature_index*num_anchors[tile_index] + anchor_index][3];
                float py2 = priors[prior_base_index[tile_index] + feature_index*num_anchors[tile_index] + anchor_index][2];

                float pw = px2-px1;
                float ph = py2-py1;
                float pcx = px1 + pw*0.5f;
                float pcy = py1 + ph*0.5f;

                float pred_cx = pcx + x * pw;
                float pred_cy = pcy + y * ph;
                float pred_w = w * pw;
                float pred_h = h * ph;

                auto b1_index = feature_index * feature_stride_for_boxes + anchor_index * anchor_stride_for_boxes + 0 * point_stride_for_boxes + 4*output_index_stride_map_for_boxes[tile_index];
                auto b2_index = feature_index * feature_stride_for_boxes + anchor_index * anchor_stride_for_boxes + 1 * point_stride_for_boxes + 4*output_index_stride_map_for_boxes[tile_index];
                auto b3_index = feature_index * feature_stride_for_boxes + anchor_index * anchor_stride_for_boxes + 2 * point_stride_for_boxes + 4*output_index_stride_map_for_boxes[tile_index];
                auto b4_index = feature_index * feature_stride_for_boxes + anchor_index * anchor_stride_for_boxes + 3 * point_stride_for_boxes + 4*output_index_stride_map_for_boxes[tile_index];

                boxes[b1_index] = pred_cx - pred_w * 0.5f;
                boxes[b2_index] = pred_cy - pred_h * 0.5f;
                boxes[b3_index] = pred_cx + pred_w * 0.5f;
                boxes[b4_index] = pred_cy + pred_h * 0.5f;
/*
				assert(b1_idx == b_counter);
                boxes[u_counter++] = pred_cx - pred_w * 0.5f;
                boxes[b_counter++] = pred_cy - pred_h * 0.5f;
                boxes[b_counter++] = pred_cx + pred_w * 0.5f;
                boxes[b_counter++] = pred_cy + pred_h * 0.5f;
                */
            }
        }
    }, *output_lowering[output_index]);

        //timing.checkpoint("NMS");

        thread_local std::array<std::vector<std::tuple<float, int>>, num_classes> picked;
        filter_results(buffers, boxes, picked);
#ifdef DEBUG
        std::cout << "picked: " << picked.size() << '\n';
#endif

        //timing.checkpoint("build result for loadgen");

        results.reserve(7*picked.size());
        for(int idx = 0; idx < num_classes; ++idx) {
            for(auto& [score, boxes_idx]: picked[idx]) {
                //results.push_back(samples[i].index);
                results.push_back(result_index);
                results.push_back(boxes[boxes_idx*4+1]);
                results.push_back(boxes[boxes_idx*4+0]);
                results.push_back(boxes[boxes_idx*4+3]);
                results.push_back(boxes[boxes_idx*4+2]);
                results.push_back(score);
                results.push_back(idx);
            }
        }

        //timing.finish();
    }
  void filter_results(
      unsigned char** buffers,
      std::array<float, 1917*4>& boxes,
      std::array<std::vector<std::tuple<float, int>>, num_classes>& picked, 
      float score_threshold = 0.3f, float nms_threshold = 0.6f
  ) {

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

      std::array<int, 6> output_index_stride_map_for_boxes;
      output_index_stride_map_for_boxes[0] = 0;
      output_index_stride_map_for_boxes[1] = 1083;
      output_index_stride_map_for_boxes[2] = 1683;
      output_index_stride_map_for_boxes[3] = 1833;
      output_index_stride_map_for_boxes[4] = 1887;
      output_index_stride_map_for_boxes[5] = 1911;


#pragma omp parallel for num_threads(16)
      for(int class_index = 1; class_index < num_classes; ++class_index) {
          thread_local std::vector<std::tuple<float, int>> filtered;
          filtered.reserve(1917);
          filtered.clear();

          for(int output_index = 0; output_index < 6; output_index++) 
              std::visit([&](auto info){
                int tile_index = output_index;
            
              for (int anchor_index = 0; anchor_index < num_anchors[tile_index]; ++anchor_index) {
                      for(int f_y = 0; f_y < feature_map_shapes[tile_index]; f_y++) 
                      for(int f_x = 0; f_x < feature_map_shapes[tile_index]; f_x++) {

                      int score_index = info.index(anchor_index*num_classes+class_index, f_y, f_x);
                      auto q = (unsigned char)buffers[tile_index][score_index];
                      auto score = output_dequantization_tables[tile_index][q];
                      if (score > score_threshold) {
                          filtered.emplace_back(score, 
                            anchor_index*feature_map_shapes[tile_index]*feature_map_shapes[tile_index] + 
                            f_y*feature_map_shapes[tile_index] + 
                            f_x + 
                            output_index_stride_map_for_boxes[tile_index]);
                      }
                  }
              } 
          }, *output_lowering[output_index]);

          // look through boxes in the descending order
          std::sort(filtered.begin(), filtered.end(), [&](auto& l, auto& r) {
              auto l_score = std::get<0>(l);
              auto r_score = std::get<0>(r);

              return l_score > r_score;
          });

          //thread_local std::vector<std::tuple<float, int>> l_picked;
          std::vector<std::tuple<float, int>>& l_picked = picked[class_index];;
          l_picked.reserve(1917);
          l_picked.clear();
          for(auto& [l_score, l_boxes_index] :filtered) {

              bool can_pick = true;
              for(auto p = 0; p < l_picked.size(); ++p) {
                  auto& [r_score, r_boxes_index] = l_picked[p];
                  if (iou(&boxes[l_boxes_index*4], &boxes[r_boxes_index*4]) > nms_threshold) {
                      can_pick = false;
                      break;
                  }
              }
              if (can_pick) {
#ifdef DEBUG
                  //std::cout 
                  //<< boxes[l_boxes_index*4+1] << '\t'
                  //<< boxes[l_boxes_index*4+0] << '\t'
                  //<< boxes[l_boxes_index*4+3] << '\t'
                  //<< boxes[l_boxes_index*4+2] << '\t'
                  //<< l_score << '\n';
#endif
                  l_picked.emplace_back(l_score, l_boxes_index);
              }
          }
          //std::swap(picked[class_index],l_picked);
      }
  }

};

class FuriosaBasicSUT : public mlperf::SystemUnderTest, SSDMoblienetReporter {
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

  FuriosaBasicSUT() 
  {
    // Start with some large value so that we don't reallocate memory.
    initResponse(1);

    std::ifstream inf("mlcommons_ssd_mobilenet_v1_int8.enf", std::ios::binary);

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
    input_lowering = TensorDescToLoweringInfo(input_desc, 3, 300, 300);
    if (input_lowering) {
        std::cout << "Using pre-lowered input.\n";
        INPUT_SIZE = std::visit([&](auto info){ return info.lowered_size(); }, *input_lowering);
    }

    auto output_desc =  nux_output_desc(model, 0);
    output_lowering[0] = TensorDescToLoweringInfo(output_desc, 91*3, 19, 19);
    output_desc =  nux_output_desc(model, 1);
    output_lowering[1] = TensorDescToLoweringInfo(output_desc, 91*6, 10, 10);
    output_desc =  nux_output_desc(model, 2);
    output_lowering[2] = TensorDescToLoweringInfo(output_desc, 91*6, 5, 5);
    output_desc =  nux_output_desc(model, 3);
    output_lowering[3] = TensorDescToLoweringInfo(output_desc, 91*6, 3, 3);
    output_desc =  nux_output_desc(model, 4);
    output_lowering[4] = TensorDescToLoweringInfo(output_desc, 91*6, 2, 2);
    output_desc =  nux_output_desc(model, 5);
    output_lowering[5] = TensorDescToLoweringInfo(output_desc, 91*6, 1, 1);
    output_desc =  nux_output_desc(model, 6);
    output_lowering[6] = TensorDescToLoweringInfo(output_desc, 4*3, 19, 19);
    output_desc =  nux_output_desc(model, 7);
    output_lowering[7] = TensorDescToLoweringInfo(output_desc, 4*6, 10, 10);
    output_desc =  nux_output_desc(model, 8);
    output_lowering[8] = TensorDescToLoweringInfo(output_desc, 4*6, 5, 5);
    output_desc =  nux_output_desc(model, 9);
    output_lowering[9] = TensorDescToLoweringInfo(output_desc, 4*6, 3, 3);
    output_desc =  nux_output_desc(model, 10);
    output_lowering[10] = TensorDescToLoweringInfo(output_desc, 4*6, 2, 2);
    output_desc =  nux_output_desc(model, 11);
    output_lowering[11] = TensorDescToLoweringInfo(output_desc, 4*6, 1, 1);

    for(int i = 0; i < 12; i ++) {
        if (output_lowering[i])
            OUTPUT_SIZE[i] = std::visit(
                [&](auto info){
                    return info.lowered_size();
                }, *output_lowering[i]);
    }

#ifdef ASYNC
    mCompleteThreads.push_back(std::thread([this]{CompleteThread();}));
#endif

    // prepare priors

  }

  std::vector<std::thread> mCompleteThreads;

  ~FuriosaBasicSUT() override {
#ifdef ASYNC
      nux_completion_queue_destroy(mQueue);
      nux_async_session_destroy(mSession);
#else
      nux_session_destroy(mSession);
#endif
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
        //std::cout << "Issuing: " << samples[i].index << ' ' << i << '/' << n << ' ' << data.size() << '\n';
        //tensor_set_buffer(tensor0, (nux_buffer_t)dummy.data(), dummy.size(), nullptr);
        auto err = tensor_set_buffer(tensor0, (nux_buffer_t)data, INPUT_SIZE, nullptr);

#ifdef ASYNC
        nux_async_session_run(mSession, (nux_context_t)samples[i].id, mInputs);
#else
        err = nux_session_run(mSession, mInputs, mOutputs);
        if (err != 0) {
            std::cout << "Error: " << err << '\n';
        }


        nux_buffer_t buffers[12];
        nux_buffer_len_t lens[12];
      for (int tile_index = 0; tile_index < 12; ++tile_index) {
            auto result = nux_tensor_array_get(mOutputs, tile_index);
            tensor_get_buffer(result, &buffers[tile_index], &lens[tile_index]);
      }

        std::vector<float> results;
        results.clear();
        post_inference(samples[i].index, buffers, results);

        mResponses[0].data = (uintptr_t)&results[0];
        mResponses[0].size = results.size() * sizeof(float);
        mlperf::QuerySamplesComplete(mResponses.data(), 1);
        //nux_tensor_array_destroy(outputs);
#endif
    }

    //mlperf::QuerySamplesComplete(mResponses.data(), n);
  }

  void FlushQueries() override {}
  void ReportLatencyResults(
      const std::vector<mlperf::QuerySampleLatency>& latencies_ns) override{};

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

class FuriosaQueueSUT : public mlperf::SystemUnderTest, SSDMoblienetReporter {
 public:
 std::vector<nux_session_t> mSessions;
  FuriosaQueueSUT(int numCompleteThreads, int maxSize) {
    std::ifstream inf("mlcommons_ssd_mobilenet_v1_int8_batch8.enf", std::ios::binary);
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
                input_lowering = TensorDescToLoweringInfo(input_desc, 3, 300, 300);
                if (input_lowering) {
                    std::cout << "Using pre-lowered input.\n";
                    INPUT_SIZE = std::visit([&](auto info){ return info.lowered_size(); }, *input_lowering);
                }

                auto output_desc =  nux_output_desc(model, 0);
                output_lowering[0] = TensorDescToLoweringInfo(output_desc, 91*3, 19, 19);
                output_desc =  nux_output_desc(model, 1);
                output_lowering[1] = TensorDescToLoweringInfo(output_desc, 91*6, 10, 10);
                output_desc =  nux_output_desc(model, 2);
                output_lowering[2] = TensorDescToLoweringInfo(output_desc, 91*6, 5, 5);
                output_desc =  nux_output_desc(model, 3);
                output_lowering[3] = TensorDescToLoweringInfo(output_desc, 91*6, 3, 3);
                output_desc =  nux_output_desc(model, 4);
                output_lowering[4] = TensorDescToLoweringInfo(output_desc, 91*6, 2, 2);
                output_desc =  nux_output_desc(model, 5);
                output_lowering[5] = TensorDescToLoweringInfo(output_desc, 91*6, 1, 1);
                output_desc =  nux_output_desc(model, 6);
                output_lowering[6] = TensorDescToLoweringInfo(output_desc, 4*3, 19, 19);
                output_desc =  nux_output_desc(model, 7);
                output_lowering[7] = TensorDescToLoweringInfo(output_desc, 4*6, 10, 10);
                output_desc =  nux_output_desc(model, 8);
                output_lowering[8] = TensorDescToLoweringInfo(output_desc, 4*6, 5, 5);
                output_desc =  nux_output_desc(model, 9);
                output_lowering[9] = TensorDescToLoweringInfo(output_desc, 4*6, 3, 3);
                output_desc =  nux_output_desc(model, 10);
                output_lowering[10] = TensorDescToLoweringInfo(output_desc, 4*6, 2, 2);
                output_desc =  nux_output_desc(model, 11);
                output_lowering[11] = TensorDescToLoweringInfo(output_desc, 4*6, 1, 1);

                for(int i = 0; i < 12; i ++) {
                    if (output_lowering[i])
                        OUTPUT_SIZE[i] = std::visit(
                            [&](auto info){
                                return info.lowered_size();
                            }, *output_lowering[i]);
                }

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
 std::atomic<int> mRunCount{0};
  void CompleteThread(int threadIdx) {
    auto& responses = mResponses[threadIdx];
    size_t maxSize{responses.size()};
    auto& session = mSessions[threadIdx];
    std::unique_ptr<unsigned char[]> inputs;
    inputs.reset((unsigned char*)std::aligned_alloc(64, INPUT_SIZE*maxSize));

    nux_model_t model = nux_session_get_model(session);

    auto input_tensor = nux_tensor_array_create_inputs(model);
    auto output_tensor = nux_tensor_array_allocate_outputs(model);

    size_t actualSize{0};
    std::vector<std::vector<float>> results;
    results.resize(maxSize);
    for(int i = 0; i < maxSize; i ++) {
        results[i].reserve(200*7);
    }

    Timing timing_inside;

    auto start_time = std::chrono::high_resolution_clock::now();

    while (true) {
        std::unique_lock<std::mutex> lck(mMtx);
        mCondVar.wait(lck, [&]() { return mWorkQueue && mWorkQueue->size() != mQueueTail || mDone; });

        if (mDone) {
            break;
        }

        auto* requests = &(*mWorkQueue)[mQueueTail];
        actualSize = std::min(maxSize, mWorkQueue->size() - mQueueTail);

        timing_inside.start();
        timing_inside.checkpoint("deque");

        mQueueTail += actualSize;
        for (int i = 0; i < actualSize; i++) {
            responses[i].id = requests[i].id;
        }
        mRunCount ++;
        if (mWorkQueue->size() == mQueueTail) {
            mWorkQueue = nullptr;
        }
        lck.unlock();
        mCondVar.notify_one();

        timing_inside.checkpoint("build input");

        for(int i = 0; i < actualSize; i ++) {
            int index = requests[i].index;
            auto* data = images[index].get();
            memcpy(&inputs[i*INPUT_SIZE], data, INPUT_SIZE);
        }

        timing_inside.checkpoint("inference");
        auto tensor0 = nux_tensor_array_get(input_tensor, 0);
        tensor_set_buffer(tensor0, (nux_buffer_t)&inputs[0], maxSize * INPUT_SIZE, nullptr);
        timing_inside.checkpoint("inference2");
        auto err = nux_session_run(session, input_tensor, output_tensor);
        if (err != 0) {
            std::cout << "Error: " << err << '\n';
        }

        timing_inside.checkpoint("read buffer");

        unsigned char* buffers[12];
        nux_buffer_len_t lens[12];
        for (int tile_index = 0; tile_index < 12; ++tile_index) {
            auto result = nux_tensor_array_get(output_tensor, tile_index);
            tensor_get_buffer(result, &buffers[tile_index], &lens[tile_index]);
        }
        timing_inside.checkpoint("postprocess");
#pragma omp parallel for
        for(int i = 0; i < actualSize; i ++) {
            results[i].clear();

            unsigned char* adjusted_buffers[12];
            for(int j = 0; j < 12; j++) {
                adjusted_buffers[j]=buffers[j]+i*OUTPUT_SIZE[j];
            }

            post_inference(requests[i].index, adjusted_buffers, results[i]);

            responses[i].data = (uintptr_t)&results[i][0];
            responses[i].size = results[i].size() * sizeof(float);
        }
        mlperf::QuerySamplesComplete(responses.data(), actualSize);
        timing_inside.finish();
    }
  }
  void initResponse(int numCompleteThreads, int size) {
      mResponses.resize(numCompleteThreads);
      for(int i = 0; i < numCompleteThreads; i ++) {
          mResponses[i].resize(size);
      }
  }
  std::string mName{"FuriosaQueueSUT"};
  std::vector<std::vector<mlperf::QuerySampleResponse>> mResponses;
  std::vector<std::thread> mThreads;
  const std::vector<mlperf::QuerySample>* mWorkQueue{nullptr};
  size_t mQueueTail;
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

struct Context {
    mlperf::ResponseId *ids;
    float *indices;
    nux_tensor_array_t input;
    nux_tensor_array_t output;
    unsigned char *buffer;
    int len;
};

class StreamSession : public SSDMoblienetReporter {
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
    float *mIndices;
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
        auto &model = mModel;
        nux_async_session_get_nux_handle(mSession, &mNux);

#ifdef DEBUG
        std::cout << "Prepare first submission" << std::endl;
#endif
        mInput = nux_tensor_array_create_inputs(mModel);
        mOutput = nux_tensor_array_allocate_outputs(mModel);
        mContext = (Context *) malloc(sizeof(Context));
        mIds = (mlperf::ResponseId *) malloc(sizeof(mlperf::ResponseId) * mModelBatch);
        mIndices = (float *) malloc(sizeof(float) * mModelBatch);
        mBuffer = (unsigned char*) std::aligned_alloc(64, INPUT_SIZE * mModelBatch);

        auto input_desc = nux_input_desc(model, 0);
        input_lowering = TensorDescToLoweringInfo(input_desc, 3, 300, 300);
        if (input_lowering) {
            std::cout << "Using pre-lowered input.\n";
            INPUT_SIZE = std::visit([&](auto info){ return info.lowered_size(); }, *input_lowering);
        }

        auto output_desc =  nux_output_desc(model, 0);
        output_lowering[0] = TensorDescToLoweringInfo(output_desc, 91*3, 19, 19);
        output_desc =  nux_output_desc(model, 1);
        output_lowering[1] = TensorDescToLoweringInfo(output_desc, 91*6, 10, 10);
        output_desc =  nux_output_desc(model, 2);
        output_lowering[2] = TensorDescToLoweringInfo(output_desc, 91*6, 5, 5);
        output_desc =  nux_output_desc(model, 3);
        output_lowering[3] = TensorDescToLoweringInfo(output_desc, 91*6, 3, 3);
        output_desc =  nux_output_desc(model, 4);
        output_lowering[4] = TensorDescToLoweringInfo(output_desc, 91*6, 2, 2);
        output_desc =  nux_output_desc(model, 5);
        output_lowering[5] = TensorDescToLoweringInfo(output_desc, 91*6, 1, 1);
        output_desc =  nux_output_desc(model, 6);
        output_lowering[6] = TensorDescToLoweringInfo(output_desc, 4*3, 19, 19);
        output_desc =  nux_output_desc(model, 7);
        output_lowering[7] = TensorDescToLoweringInfo(output_desc, 4*6, 10, 10);
        output_desc =  nux_output_desc(model, 8);
        output_lowering[8] = TensorDescToLoweringInfo(output_desc, 4*6, 5, 5);
        output_desc =  nux_output_desc(model, 9);
        output_lowering[9] = TensorDescToLoweringInfo(output_desc, 4*6, 3, 3);
        output_desc =  nux_output_desc(model, 10);
        output_lowering[10] = TensorDescToLoweringInfo(output_desc, 4*6, 2, 2);
        output_desc =  nux_output_desc(model, 11);
        output_lowering[11] = TensorDescToLoweringInfo(output_desc, 4*6, 1, 1);

        for(int i = 0; i < 12; i ++) {
            if (output_lowering[i])
                OUTPUT_SIZE[i] = std::visit(
                    [&](auto info){
                        return info.lowered_size();
                    }, *output_lowering[i]);
        }

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
        free(mIndices);
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
            mIndices[0] = samples[begin].index;
            auto tensor = nux_tensor_array_get(mInput, 0);
            auto* data = images[samples[begin].index].get();
            auto err = tensor_set_buffer(tensor, (nux_buffer_t) data, INPUT_SIZE, nullptr);
            if (err != 0) {
                std::cout << "Error: " << err << '\n';
            }
        } else {
            for(int i = 0; i < len ; i ++) {
                mIds[i] = samples[begin + i].id;
                mIndices[i] = samples[begin + i].index;
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
        mContext->indices = mIndices;
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
        mIndices = (float *) malloc(sizeof(float) * mModelBatch);
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

        std::vector<std::vector<float>> results;
        results.resize(mModelBatch);
        for(int i = 0; i < mModelBatch; i ++) {
            results[i].reserve(200*7);
        }

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


            unsigned char* buffers[12];
            nux_buffer_len_t lens[12];
            for (int tile_index = 0; tile_index < 12; ++tile_index) {
                auto result = nux_tensor_array_get(ctx->output, tile_index);
                tensor_get_buffer(result, &buffers[tile_index], &lens[tile_index]);
            }

#pragma omp parallel for
            for(int i = 0; i < ctx->len; i ++) {
                results[i].clear();

                unsigned char* adjusted_buffers[12];
                for(int j = 0; j < 12; j++) {
                    adjusted_buffers[j]=buffers[j]+i*OUTPUT_SIZE[j];
                }

                post_inference(ctx->indices[i], adjusted_buffers, results[i]);

                responses[i].id = ctx->ids[i];
                responses[i].data = (uintptr_t)&results[i][0];
                responses[i].size = results[i].size() * sizeof(float);
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

class BlockingSession : public SSDMoblienetReporter {
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
        auto &model = mModel;
        nux_session_get_nux_handle(mSession, &mNux);

        auto input_desc = nux_input_desc(model, 0);
        input_lowering = TensorDescToLoweringInfo(input_desc, 3, 300, 300);
        if (input_lowering) {
            std::cout << "Using pre-lowered input.\n";
            INPUT_SIZE = std::visit([&](auto info){ return info.lowered_size(); }, *input_lowering);
        }

        auto output_desc =  nux_output_desc(model, 0);
        output_lowering[0] = TensorDescToLoweringInfo(output_desc, 91*3, 19, 19);
        output_desc =  nux_output_desc(model, 1);
        output_lowering[1] = TensorDescToLoweringInfo(output_desc, 91*6, 10, 10);
        output_desc =  nux_output_desc(model, 2);
        output_lowering[2] = TensorDescToLoweringInfo(output_desc, 91*6, 5, 5);
        output_desc =  nux_output_desc(model, 3);
        output_lowering[3] = TensorDescToLoweringInfo(output_desc, 91*6, 3, 3);
        output_desc =  nux_output_desc(model, 4);
        output_lowering[4] = TensorDescToLoweringInfo(output_desc, 91*6, 2, 2);
        output_desc =  nux_output_desc(model, 5);
        output_lowering[5] = TensorDescToLoweringInfo(output_desc, 91*6, 1, 1);
        output_desc =  nux_output_desc(model, 6);
        output_lowering[6] = TensorDescToLoweringInfo(output_desc, 4*3, 19, 19);
        output_desc =  nux_output_desc(model, 7);
        output_lowering[7] = TensorDescToLoweringInfo(output_desc, 4*6, 10, 10);
        output_desc =  nux_output_desc(model, 8);
        output_lowering[8] = TensorDescToLoweringInfo(output_desc, 4*6, 5, 5);
        output_desc =  nux_output_desc(model, 9);
        output_lowering[9] = TensorDescToLoweringInfo(output_desc, 4*6, 3, 3);
        output_desc =  nux_output_desc(model, 10);
        output_lowering[10] = TensorDescToLoweringInfo(output_desc, 4*6, 2, 2);
        output_desc =  nux_output_desc(model, 11);
        output_lowering[11] = TensorDescToLoweringInfo(output_desc, 4*6, 1, 1);

        for(int i = 0; i < 12; i ++) {
            if (output_lowering[i])
                OUTPUT_SIZE[i] = std::visit(
                    [&](auto info){
                        return info.lowered_size();
                    }, *output_lowering[i]);
        }

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


class FuriosaSingleThreadSUT : public mlperf::SystemUnderTest {
public:
#ifdef ASYNC
    std::unique_ptr<StreamSession> mSession;
#else
    std::unique_ptr<BlockingSession> mSession;
#endif

    FuriosaSingleThreadSUT(int numWorkers = 12) {
#ifdef DEBUG
        std::cout << "Constructing FuriosaSingleThreadSUT" << std::endl;
#endif
#ifdef ASYNC
        mSession.reset(new StreamSession(
			numWorkers, 8, "npu0pe0-1", "mlcommons_resnet50_v1.5_int8.enf"));
        
#else
        mSession.reset(new BlockingSession());
#endif
    }

    ~FuriosaSingleThreadSUT() {
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
    if (getenv("META")) {
        META_PATH = getenv("META");
    }

  //assert(argc >= 2 && "Need to pass in at least one argument: target_qps");
  int target_qps = 100;
  bool useQueue{false};
  int numCompleteThreads{4};
  int maxSize{1};
  bool server_coalesce_queries{false};
  int num_issue_threads{0};

  bool isAccuracyRun = false;
  int num_samples = 5000;
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
  if (num_samples > 5000)
	  num_samples = 5000;
  if (isAccuracyRun) {
	  std::cout << "Accuracy mode." << '\n';
  } else {
	  std::cout << "Performance mode." << '\n';
  }

  QSL qsl(num_samples);
  std::unique_ptr<mlperf::SystemUnderTest> sut;

  // Configure the test settings
  testSettings.FromConfig("../common/mlperf.conf", "ssd-mobilenet", arg_scenario == mlperf::TestScenario::SingleStream ? "SingleStream" : "Offline");
  testSettings.FromConfig("../common/user.conf", "ssd-mobilenet", arg_scenario == mlperf::TestScenario::SingleStream ? "SingleStream" : "Offline");
  testSettings.scenario = arg_scenario;
  testSettings.mode = mlperf::TestMode::PerformanceOnly;
  testSettings.single_stream_expected_latency_ns = 460000;
  if (isAccuracyRun)
      testSettings.mode = mlperf::TestMode::AccuracyOnly;
  //testSettings.server_target_qps = target_qps;
  //testSettings.server_target_latency_ns = 500000;  // 0.5ms
  //testSettings.server_target_latency_percentile = 0.99;

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
      sut.reset(new FuriosaBasicSUT());
  } else {
      sut.reset(new FuriosaQueueSUT(2, 8));
  }

  // Start test
  std::cout << "Start test..." << std::endl;
  mlperf::StartTest(sut.get(), &qsl, testSettings, logSettings);
  std::cout << "Test done. Clean up SUT..." << std::endl;
  sut.reset();
  std::cout << "Done!" << std::endl;
  return 0;
}
