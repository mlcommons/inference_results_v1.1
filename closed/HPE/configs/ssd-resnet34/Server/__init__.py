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


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx4(BenchmarkConfiguration):
    system = System("A100-SXM-80GB", Architecture.Ampere, 4)
    active_sms = 100
    gpu_copy_streams = 4
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/coco/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/coco/val2017/SSDResNet34/int8_linear"
    use_deque_limit = True
    use_graphs = False
    deque_timeout_usec = 30000
    gpu_batch_size = 8
    gpu_inference_streams = 4
    server_target_qps = 3700
    start_from_device = False
    use_cuda_thread_per_device = True
    scenario = Scenario.Server
    benchmark = Benchmark.SSDResNet34


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx4_Triton(BenchmarkConfiguration):
    system = System("A100-SXM-80GB", Architecture.Ampere, 4)
    active_sms = 100
    gpu_copy_streams = 4
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/coco/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/coco/val2017/SSDResNet34/int8_linear"
    use_deque_limit = True
    use_graphs = False
    deque_timeout_usec = 30000
    gpu_batch_size = 8
    gpu_inference_streams = 4
    server_target_qps = 3550
    use_cuda_thread_per_device = True
    scenario = Scenario.Server
    benchmark = Benchmark.SSDResNet34
    instance_group_count = 4
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx8(BenchmarkConfiguration):
    system = System("A100-SXM-80GB", Architecture.Ampere, 8)
    active_sms = 100
    gpu_copy_streams = 4
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/coco/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/coco/val2017/SSDResNet34/int8_linear"
    use_deque_limit = True
    use_graphs = False
    deque_timeout_usec = 30000
    gpu_batch_size = 8
    gpu_inference_streams = 4
    server_target_qps = 7650
    start_from_device = False
    use_cuda_thread_per_device = True
    scenario = Scenario.Server
    benchmark = Benchmark.SSDResNet34


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx8_Triton(BenchmarkConfiguration):
    system = System("A100-SXM-80GB", Architecture.Ampere, 8)
    active_sms = 100
    gpu_copy_streams = 4
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/coco/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/coco/val2017/SSDResNet34/int8_linear"
    use_deque_limit = True
    use_graphs = False
    deque_timeout_usec = 30000
    gpu_batch_size = 8
    gpu_inference_streams = 4
    server_target_qps = 7100
    use_cuda_thread_per_device = True
    scenario = Scenario.Server
    benchmark = Benchmark.SSDResNet34
    instance_group_count = 4
    use_triton = True
