#pragma once

#include <ATen/core/ivalue.h>
#include <string>
#include <torch/script.h>
#include <ATen/Parallel.h>
#include "bert_qsl.hpp"

namespace models {
//
// Functionality:
//   1. Model load&adjust
//
class TorchModel {
public:
  TorchModel (const std::string filename) : model_(torch::jit::load(filename)) {
    model_.eval();
    socket_model_[0] = model_;
    socket_model_[1] = model_.clone();
  }

  TorchModel ();

  void load(const std::string filename) {
    model_ = torch::jit::load(filename);
  }

  template <typename... Args>
  at::IValue inference(Args&&... args) {
    return model_.forward(std::forward<Args>(args)...);
  }

  template <typename... Args>
  at::IValue inference_at(int socket, Args&&... args) {
    return socket_model_[socket].forward(std::forward<Args>(args)...);
  }

private:
  torch::jit::script::Module model_;
  torch::jit::script::Module socket_model_[2];
};

}
