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
    gpu_copy_streams = 4								
    input_dtype = "int8"								
    input_format = "linear"								
    map_path = "data_maps/coco/val_map.txt"								
    precision = "int8"								
    tensor_path = "${PREPROCESSED_DATA_DIR}/coco/val2017/SSDResNet34/int8_linear"
    use_deque_limit = True								
    use_graphs = False								
    deque_timeout_usec = 38041								
    gpu_batch_size = 34								
    gpu_inference_streams = 1								
    server_target_qps = 8950
    use_cuda_thread_per_device = True								
    scenario = Scenario.Server								
    benchmark = Benchmark.SSDResNet34								
    start_from_device = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class DSS8440_A30x8(BenchmarkConfiguration):
    system = System("DSS8440_A30", Architecture.Ampere, 8)
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
    gpu_inference_streams = 2
    server_target_qps = 3572
    use_cuda_thread_per_device = True
    scenario = Scenario.Server
    benchmark = Benchmark.SSDResNet34
    start_from_device = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class R750xa_A100_PCIE_40GBx4_MaxQ(BenchmarkConfiguration):
    system = System("R750xa_A100-PCIE-40GB", Architecture.Ampere, 4)
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
    gpu_inference_streams = 2
    server_target_qps = 630*4
    use_cuda_thread_per_device = True
    scenario = Scenario.Server
    benchmark = Benchmark.SSDResNet34
    start_from_device = True
    power_limit = 175


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750xa_A100_PCIE_80GBx4(BenchmarkConfiguration):
    system = System("R750xa_A100-PCIE-80GB", Architecture.Ampere, 4)
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
    gpu_batch_size = 16
    gpu_inference_streams = 2
    server_target_qps = 3250
    use_cuda_thread_per_device = True
    scenario = Scenario.Server
    benchmark = Benchmark.SSDResNet34
    start_from_device = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750xa_A100_PCIE_80GBx4_Triton(BenchmarkConfiguration):
    system = System("R750xa_A100-PCIE-80GB", Architecture.Ampere, 4)
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
    gpu_batch_size = 16
    gpu_inference_streams = 2
    server_target_qps = 2850
    use_cuda_thread_per_device = True
    scenario = Scenario.Server
    benchmark = Benchmark.SSDResNet34
    instance_group_count = 4
    use_triton = True
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
    gpu_copy_streams = 2
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/coco/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/coco/val2017/SSDResNet34/int8_linear"
    use_deque_limit = True
    use_graphs = False
    deque_timeout_usec = 1000
    gpu_batch_size = 2
    gpu_inference_streams = 2
    server_target_qps = 3500
    start_from_device = True
    use_cuda_thread_per_device = False
    scenario = Scenario.Server
    benchmark = Benchmark.SSDResNet34


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
    gpu_copy_streams = 2
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/coco/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/coco/val2017/SSDResNet34/int8_linear"
    use_deque_limit = True
    use_graphs = False
    deque_timeout_usec = 1000
    gpu_batch_size = 2
    gpu_inference_streams = 2
    server_target_qps = 2900
    start_from_device = True
    use_cuda_thread_per_device = False
    scenario = Scenario.Server
    benchmark = Benchmark.SSDResNet34
    instance_group_count = 2
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R7525_A100_PCIE_40GBx3(BenchmarkConfiguration):
    system = System("R7525_A100-PCIE-40GB", Architecture.Ampere, 3)
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
    gpu_batch_size = 11
    gpu_inference_streams = 3
    server_target_qps = 2389
    use_cuda_thread_per_device = True
    scenario = Scenario.Server
    benchmark = Benchmark.SSDResNet34
    start_from_device = True
    numa_config = "0:0-31&1-2:32-63"


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R7525_A30x3(BenchmarkConfiguration):
    system = System("R7525_A30", Architecture.Ampere, 3)
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
    gpu_inference_streams = 2
    server_target_qps = 446.59999999999997*3
    use_cuda_thread_per_device = True
    scenario = Scenario.Server
    benchmark = Benchmark.SSDResNet34


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE2420_A10x2(BenchmarkConfiguration):
    system = System("XE2420_A10", Architecture.Ampere, 2)
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
    server_target_qps = 570
    use_cuda_thread_per_device = True
    scenario = Scenario.Server
    benchmark = Benchmark.SSDResNet34


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class XE2420_A10x2_MaxQ(XE2420_A10x2):
    pass


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8545_A100_SXM_80GBx4(BenchmarkConfiguration):
    system = System("XE8545_A100-SXM-80GB", Architecture.Ampere, 4)
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
    server_target_qps = 3960
    start_from_device = True
    use_cuda_thread_per_device = True
    scenario = Scenario.Server
    benchmark = Benchmark.SSDResNet34


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8545_A100_SXM_80GBx4_Triton(BenchmarkConfiguration):
    system = System("XE8545_A100-SXM-80GB", Architecture.Ampere, 4)
    active_sms = 100
    gpu_copy_streams = 4
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/coco/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/coco/val2017/SSDResNet34/int8_linear"
    use_deque_limit = True
    use_graphs = False
    deque_timeout_usec = 32205
    gpu_batch_size = 9
    gpu_inference_streams = 9
    server_target_qps = 3900
    use_cuda_thread_per_device = True
    scenario = Scenario.Server
    benchmark = Benchmark.SSDResNet34
    instance_group_count = 4
    use_triton = True
    start_from_device = True


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
    gpu_copy_streams = 2
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/coco/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/coco/val2017/SSDResNet34/int8_linear"
    use_deque_limit = True
    use_graphs = False
    deque_timeout_usec = 1439
    gpu_batch_size = 2
    gpu_inference_streams = 1
    server_target_qps = 3466
    start_from_device = True
    use_cuda_thread_per_device = False
    scenario = Scenario.Server
    benchmark = Benchmark.SSDResNet34
    instance_group_count = 2
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR12_A10x2(BenchmarkConfiguration):
    system = System("XR12_A10", Architecture.Ampere, 2)
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
    server_target_qps = 550
    use_cuda_thread_per_device = True
    scenario = Scenario.Server
    benchmark = Benchmark.SSDResNet34


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIex3(BenchmarkConfiguration):
    system = System("R7525_vA100-PCIE-40GB", Architecture.Ampere, 3)
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
    gpu_inference_streams = 2
    server_target_qps = 2310.0
    use_cuda_thread_per_device = True
    scenario = Scenario.Server
    benchmark = Benchmark.SSDResNet34
