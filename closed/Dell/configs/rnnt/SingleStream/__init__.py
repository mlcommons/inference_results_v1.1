# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
sys.path.insert(0, os.getcwd())

from code.common.constants import Benchmark, Scenario
from code.common.system_list import System, Architecture, MIGConfiguration, MIGSlice
from configs.configuration import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE2420_A10x1(BenchmarkConfiguration):
    system = System("XE2420_A10", Architecture.Ampere, 1)
    disable_encoder_plugin = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "fp16"
    input_format = "linear"
    map_path = "data_maps/rnnt_dev_clean_512/val_map.txt"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/rnnt_dev_clean_512/fp16"
    use_graphs = True
    audio_batch_size = 1
    audio_buffer_num_lines = 1
    audio_fp16_input = True
    dali_batches_issue_ahead = 1
    dali_pipeline_depth = 1
    single_stream_expected_latency_ns = 25000000
    nobatch_sorting = True
    nouse_copy_kernel = True
    num_warmups = 32
    scenario = Scenario.SingleStream
    benchmark = Benchmark.RNNT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class XE2420_A10x1_MaxQ(XE2420_A10x1):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE2420_A10x2(XE2420_A10x1):
    system = System("XE2420_A10", Architecture.Ampere, 2)
    disable_encoder_plugin = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "fp16"
    input_format = "linear"
    map_path = "data_maps/rnnt_dev_clean_512/val_map.txt"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/rnnt_dev_clean_512/fp16"
    use_graphs = True
    audio_batch_size = 1
    audio_buffer_num_lines = 1
    audio_fp16_input = True
    dali_batches_issue_ahead = 1
    dali_pipeline_depth = 1
    single_stream_expected_latency_ns = 25000000
    nobatch_sorting = True
    nouse_copy_kernel = True
    num_warmups = 32
    scenario = Scenario.SingleStream
    benchmark = Benchmark.RNNT
