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
class DSS8440_A100_PCIE_80GBx10(BenchmarkConfiguration):
    system = System("DSS8440_A100-PCIE-80GB", Architecture.Ampere, 10)
    active_sms = 100
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/imagenet/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/imagenet/ResNet50/int8_linear"
    use_deque_limit = True
    deque_timeout_usec = 6820
    gpu_batch_size = 55
    gpu_copy_streams = 6
    gpu_inference_streams = 11
    server_target_qps = 272250 
    use_cuda_thread_per_device = True
    use_graphs = True
    scenario = Scenario.Server
    benchmark = Benchmark.ResNet50
    numa_config = "0-4:0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92,94&5-9:1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59,61,63,65,67,69,71,73,75,77,79,81,83,85,87,89,91,93,95"


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class DSS8440_A30x8(BenchmarkConfiguration):
    system = System("DSS8440_A30", Architecture.Ampere, 8)
    active_sms = 100
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/imagenet/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/imagenet/ResNet50/int8_linear"
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 160
    gpu_copy_streams = 4
    gpu_inference_streams = 3
    server_target_qps = 144000
    use_cuda_thread_per_device = True
    use_graphs = True
    scenario = Scenario.Server
    benchmark = Benchmark.ResNet50
    start_from_device = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class DSS8440_A30x8_Triton(BenchmarkConfiguration):
    system = System("DSS8440_A30", Architecture.Ampere, 8)
    active_sms = 100
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/imagenet/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/imagenet/ResNet50/int8_linear"
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 160
    gpu_copy_streams = 4
    gpu_inference_streams = 3
    server_target_qps = 99000
    use_cuda_thread_per_device = True
    use_graphs = False
    scenario = Scenario.Server
    benchmark = Benchmark.ResNet50
    use_triton = True
    gather_kernel_buffer_threshold = 64
    max_queue_delay_usec = 1000
    request_timeout_usec = 8000
    start_from_device = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class R750xa_A100_PCIE_40GBx4_MaxQ(BenchmarkConfiguration):
    system = System("R750xa_A100-PCIE-40GB", Architecture.Ampere, 4)
    active_sms = 100
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/imagenet/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/imagenet/ResNet50/int8_linear"
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 64
    gpu_copy_streams = 4
    gpu_inference_streams = 3
    server_target_qps = 99670
    use_cuda_thread_per_device = True
    use_graphs = True
    scenario = Scenario.Server
    benchmark = Benchmark.ResNet50
    power_limit = 175
    start_from_device= True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750xa_A100_PCIE_80GBx4(BenchmarkConfiguration):
    system = System("R750xa_A100-PCIE-80GB", Architecture.Ampere, 4)
    active_sms = 100
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/imagenet/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/imagenet/ResNet50/int8_linear"
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 256
    gpu_copy_streams = 4
    gpu_inference_streams = 5
    server_target_qps = 135000
    use_cuda_thread_per_device = True
    use_graphs = True
    scenario = Scenario.Server
    benchmark = Benchmark.ResNet50
    start_from_device = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750xa_A100_PCIE_80GBx4_Triton(BenchmarkConfiguration):
    system = System("R750xa_A100-PCIE-80GB", Architecture.Ampere, 4)
    active_sms = 100
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/imagenet/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/imagenet/ResNet50/int8_linear"
    use_deque_limit = True
    deque_timeout_usec = 500
    gpu_batch_size = 256
    gpu_copy_streams = 4
    gpu_inference_streams = 3
    server_target_qps = 126000
    use_cuda_thread_per_device = True
    use_graphs = False
    scenario = Scenario.Server
    benchmark = Benchmark.ResNet50
    use_triton = True
    max_queue_delay_usec = 1000
    request_timeout_usec = 2000
    start_from_device = True



@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750xa_A100_PCIE_80GB_MIG_28x1g10gb(BenchmarkConfiguration):
    _mig_configuration = MIGConfiguration({
        0: {MIGSlice(1, 10): 7},
        1: {MIGSlice(1, 10): 7},
        2: {MIGSlice(1, 10): 7},
        3: {MIGSlice(1, 10): 7},
    })
    system = System("R750xa_A100-PCIE-80GB", Architecture.Ampere, 4, mig_conf=_mig_configuration)
    active_sms = 100
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/imagenet/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/imagenet/ResNet50/int8_linear"
    use_deque_limit = True
    deque_timeout_usec = 200
    gpu_batch_size = 8
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    server_target_qps = 36000
    start_from_device = True
    use_graphs = True
    scenario = Scenario.Server
    benchmark = Benchmark.ResNet50


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750xa_A100_PCIE_80GB_MIG_28x1g10gb_Triton(BenchmarkConfiguration):
    _mig_configuration = MIGConfiguration({
        0: {MIGSlice(1, 10): 7},
        1: {MIGSlice(1, 10): 7},
        2: {MIGSlice(1, 10): 7},
        3: {MIGSlice(1, 10): 7},
    })
    system = System("R750xa_A100-PCIE-80GB", Architecture.Ampere, 4, mig_conf=_mig_configuration)
    active_sms = 100
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/imagenet/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/imagenet/ResNet50/int8_linear"
    use_deque_limit = True
    deque_timeout_usec = 10
    gpu_batch_size = 8
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    server_target_qps = 51000
    use_graphs = False
    start_from_device=True
    scenario = Scenario.Server
    benchmark = Benchmark.ResNet50
    use_triton = True
    instance_group_count = 1
    max_queue_delay_usec = 10
    preferred_batch_size = 1


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R7525_A100_PCIE_40GBx3(BenchmarkConfiguration):
    system = System("R7525_A100-PCIE-40GB", Architecture.Ampere, 3)
    active_sms = 100
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/imagenet/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/imagenet/ResNet50/int8_linear"
    use_deque_limit = True
    deque_timeout_usec = 5742
    gpu_batch_size = 205
    gpu_copy_streams = 11
    gpu_inference_streams = 9
    server_target_qps = 91250
    use_cuda_thread_per_device = True
    use_graphs = True
    scenario = Scenario.Server
    benchmark = Benchmark.ResNet50
    start_from_device=True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R7525_A30x3(BenchmarkConfiguration):
    system = System("R7525_A30", Architecture.Ampere, 3)
    active_sms = 100
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/imagenet/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/imagenet/ResNet50/int8_linear"
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 64
    gpu_copy_streams = 3
    gpu_inference_streams = 4
    server_target_qps = 47000
    use_cuda_thread_per_device = True
    use_graphs = True
    scenario = Scenario.Server
    benchmark = Benchmark.ResNet50
    start_from_device = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE2420_A10x2(BenchmarkConfiguration):
    system = System("XE2420_A10", Architecture.Ampere, 2)
    active_sms = 100
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/imagenet/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/imagenet/ResNet50/int8_linear"
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 128
    gpu_copy_streams = 4
    gpu_inference_streams = 4
    server_target_qps = 25100
    use_cuda_thread_per_device = True
    use_graphs = True
    scenario = Scenario.Server
    benchmark = Benchmark.ResNet50


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class XE2420_A10x2_MaxQ(XE2420_A10x2):
    server_target_qps = 24100
    pass


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8545_A100_SXM_80GBx4(BenchmarkConfiguration):
    system = System("XE8545_A100-SXM-80GB", Architecture.Ampere, 4)
    active_sms = 100
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/imagenet/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/imagenet/ResNet50/int8_linear"
    use_deque_limit = True
    deque_timeout_usec = 4000
    gpu_batch_size = 128
    gpu_copy_streams = 4
    gpu_inference_streams = 2
    server_target_qps = 140000
    start_from_device = True
    use_cuda_thread_per_device = True
    use_graphs = True
    scenario = Scenario.Server
    benchmark = Benchmark.ResNet50


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8545_A100_SXM_80GBx4(BenchmarkConfiguration):
    system = System("XE8545_A100-SXM-80GB", Architecture.Ampere, 4)
    active_sms = 100
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/imagenet/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/imagenet/ResNet50/int8_linear"
    use_deque_limit = True
    deque_timeout_usec = 4479
    gpu_batch_size = 106
    gpu_copy_streams = 10
    gpu_inference_streams = 10
    server_target_qps = 110000
    use_cuda_thread_per_device = True
    use_graphs = False
    use_triton = True
    start_from_device = True
    scenario = Scenario.Server
    benchmark = Benchmark.ResNet50


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8545_A100_SXM_80GB_MIG_28x1g10gb_Triton(BenchmarkConfiguration):
    _mig_configuration = MIGConfiguration({
        0: {MIGSlice(1, 10): 7},
        1: {MIGSlice(1, 10): 7},
        2: {MIGSlice(1, 10): 7},
        3: {MIGSlice(1, 10): 7},
    })
    system = System("XE8545_A100-SXM-80GB", Architecture.Ampere, 4, mig_conf=_mig_configuration)
    active_sms = 100
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/imagenet/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/imagenet/ResNet50/int8_linear"
    use_deque_limit = True
    deque_timeout_usec = 237
    gpu_batch_size = 8
    gpu_copy_streams = 11
    gpu_inference_streams = 9
    server_target_qps = 69937
    use_graphs = False
    scenario = Scenario.Server
    benchmark = Benchmark.ResNet50
    use_triton = True
    instance_group_count = 16
    max_queue_delay_usec = 500
    preferred_batch_size = 2
    start_from_device = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR12_A10x2(BenchmarkConfiguration):
    system = System("XR12_A10", Architecture.Ampere, 2)
    active_sms = 100
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/imagenet/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/imagenet/ResNet50/int8_linear"
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 128
    gpu_copy_streams = 4
    gpu_inference_streams = 4
    server_target_qps = 24500
    use_cuda_thread_per_device = True
    use_graphs = True
    scenario = Scenario.Server
    benchmark = Benchmark.ResNet50

