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
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/coco/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/coco/val2017/SSDResNet34/int8_linear"
    use_graphs = False
    gpu_batch_size = 64
    gpu_copy_streams = 4
    gpu_inference_streams = 2
    run_infer_on_copy_streams = False
    scenario = Scenario.Offline
    benchmark = Benchmark.SSDResNet34
    offline_expected_qps = 960*10
    start_from_device = True 
    numa_config = "0-4:0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92,94&5-9:1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59,61,63,65,67,69,71,73,75,77,79,81,83,85,87,89,91,93,95"


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class DSS8440_A30x8(BenchmarkConfiguration):
    system = System("DSS8440_A30", Architecture.Ampere, 8)
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/coco/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/coco/val2017/SSDResNet34/int8_linear"
    use_graphs = False
    gpu_batch_size = 32
    gpu_copy_streams = 4
    gpu_inference_streams = 2
    run_infer_on_copy_streams = False
    scenario = Scenario.Offline
    benchmark = Benchmark.SSDResNet34
    offline_expected_qps = 3760.0


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class R750xa_A100_PCIE_40GBx4_MaxQ(BenchmarkConfiguration):
    system = System("R750xa_A100-PCIE-40GB", Architecture.Ampere, 4)
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/coco/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/coco/val2017/SSDResNet34/int8_linear"
    use_graphs = False
    gpu_batch_size = 64
    gpu_copy_streams = 4
    gpu_inference_streams = 2
    run_infer_on_copy_streams = False
    scenario = Scenario.Offline
    benchmark = Benchmark.SSDResNet34
    offline_expected_qps = 960*4
    start_from_device= True
    power_limit = 175


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750xa_A100_PCIE_80GBx4(BenchmarkConfiguration):
    system = System("R750xa_A100-PCIE-80GB", Architecture.Ampere, 4)
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/coco/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/coco/val2017/SSDResNet34/int8_linear"
    use_graphs = False
    gpu_batch_size = 64
    gpu_copy_streams = 4
    gpu_inference_streams = 2
    run_infer_on_copy_streams = False
    scenario = Scenario.Offline
    benchmark = Benchmark.SSDResNet34
    offline_expected_qps = 3840.0


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750xa_A100_PCIE_80GBx4_Triton(BenchmarkConfiguration):
    system = System("R750xa_A100-PCIE-80GB", Architecture.Ampere, 4)
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/coco/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/coco/val2017/SSDResNet34/int8_linear"
    use_graphs = False
    gpu_batch_size = 64
    gpu_copy_streams = 4
    gpu_inference_streams = 2
    run_infer_on_copy_streams = False
    scenario = Scenario.Offline
    benchmark = Benchmark.SSDResNet34
    use_triton = True
    offline_expected_qps = 3840.0


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750xa_A100_PCIE_80GB_MIG_28x1g10gb(BenchmarkConfiguration):
    _mig_configuration = MIGConfiguration({
        0: {MIGSlice(1, 10): 7},
        1: {MIGSlice(1, 10): 7},
        2: {MIGSlice(1, 10): 7},
        3: {MIGSlice(1, 10): 7},
    })
    system = System("R750xa_A100-PCIE-80GB", Architecture.Ampere, 4, mig_conf=_mig_configuration)
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/coco/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/coco/val2017/SSDResNet34/int8_linear"
    use_graphs = False
    gpu_batch_size = 32
    gpu_copy_streams = 2
    gpu_inference_streams = 2
    start_from_device = True
    scenario = Scenario.Offline
    benchmark = Benchmark.SSDResNet34
    offline_expected_qps = 3700


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750xa_A100_PCIE_80GB_MIG_28x1g10gb_Triton(BenchmarkConfiguration):
    _mig_configuration = MIGConfiguration({
        0: {MIGSlice(1, 10): 7},
        1: {MIGSlice(1, 10): 7},
        2: {MIGSlice(1, 10): 7},
        3: {MIGSlice(1, 10): 7},
    })
    system = System("R750xa_A100-PCIE-80GB", Architecture.Ampere, 4, mig_conf=_mig_configuration)
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/coco/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/coco/val2017/SSDResNet34/int8_linear"
    use_graphs = False
    gpu_batch_size = 32
    gpu_copy_streams = 2
    gpu_inference_streams = 2
    start_from_device = True
    scenario = Scenario.Offline
    benchmark = Benchmark.SSDResNet34
    use_triton = True
    offline_expected_qps = 3700


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R7525_A100_PCIE_40GBx3(BenchmarkConfiguration):
    system = System("R7525_A100-PCIE-40GB", Architecture.Ampere, 3)
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/coco/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/coco/val2017/SSDResNet34/int8_linear"
    use_graphs = False
    gpu_batch_size = 36
    gpu_copy_streams = 11
    gpu_inference_streams = 2
    run_infer_on_copy_streams = False
    scenario = Scenario.Offline
    benchmark = Benchmark.SSDResNet34
    offline_expected_qps = 4206
    start_from_device = True
    numa_config = "0:0-31&1-2:32-63"


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R7525_A30x3(BenchmarkConfiguration):
    system = System("R7525_A30", Architecture.Ampere, 3)
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/coco/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/coco/val2017/SSDResNet34/int8_linear"
    use_graphs = False
    gpu_batch_size = 32
    gpu_copy_streams = 4
    gpu_inference_streams = 2
    run_infer_on_copy_streams = False
    scenario = Scenario.Offline
    benchmark = Benchmark.SSDResNet34
    offline_expected_qps = 470*3


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE2420_A10x1(BenchmarkConfiguration):
    system = System("XE2420_A10", Architecture.Ampere, 1)
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/coco/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/coco/val2017/SSDResNet34/int8_linear"
    use_graphs = False
    gpu_batch_size = 32
    gpu_copy_streams = 4
    gpu_inference_streams = 1
    scenario = Scenario.Offline
    benchmark = Benchmark.SSDResNet34
    offline_expected_qps = 1000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class XE2420_A10x1_MaxQ(XE2420_A10x1):
    pass


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE2420_A10x2(XE2420_A10x1):
    system = System("XE2420_A10", Architecture.Ampere, 2)
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/coco/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/coco/val2017/SSDResNet34/int8_linear"
    use_graphs = False
    gpu_batch_size = 32
    gpu_copy_streams = 4
    gpu_inference_streams = 1
    scenario = Scenario.Offline
    benchmark = Benchmark.SSDResNet34
    offline_expected_qps = 1000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class XE2420_A10x2_MaxQ(XE2420_A10x2):
    pass


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8545_A100_SXM_80GBx4(BenchmarkConfiguration):
    system = System("XE8545_A100-SXM-80GB", Architecture.Ampere, 4)
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/coco/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/coco/val2017/SSDResNet34/int8_linear"
    use_graphs = False
    gpu_batch_size = 64
    gpu_copy_streams = 4
    gpu_inference_streams = 2
    start_from_device = True
    run_infer_on_copy_streams = False
    scenario = Scenario.Offline
    benchmark = Benchmark.SSDResNet34
    offline_expected_qps = 4150


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8545_A100_SXM_80GBx4_Triton(BenchmarkConfiguration):
    system = System("XE8545_A100-SXM-80GB", Architecture.Ampere, 4)
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/coco/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/coco/val2017/SSDResNet34/int8_linear"
    use_graphs = False
    gpu_batch_size = 64
    gpu_copy_streams = 4
    gpu_inference_streams = 2
    run_infer_on_copy_streams = False
    scenario = Scenario.Offline
    benchmark = Benchmark.SSDResNet34
    use_triton = True
    offline_expected_qps = 4092 


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8545_A100_SXM_80GB_MIG_28x1g10gb_Triton(BenchmarkConfiguration):
    _mig_configuration = MIGConfiguration({
        0: {MIGSlice(1, 10): 7},
        1: {MIGSlice(1, 10): 7},
        2: {MIGSlice(1, 10): 7},
        3: {MIGSlice(1, 10): 7},
    })
    system = System("XE8545_A100-SXM-80GB", Architecture.Ampere, 4, mig_conf=_mig_configuration)
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/coco/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/coco/val2017/SSDResNet34/int8_linear"
    use_graphs = False
    gpu_batch_size = 32
    gpu_copy_streams = 2
    gpu_inference_streams = 2
    start_from_device = True
    scenario = Scenario.Offline
    benchmark = Benchmark.SSDResNet34
    use_triton = True
    offline_expected_qps = 3780


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR12_A10x2(BenchmarkConfiguration):
    system = System("XR12_A10", Architecture.Ampere, 2)
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/coco/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/coco/val2017/SSDResNet34/int8_linear"
    use_graphs = False
    gpu_batch_size = 32
    gpu_copy_streams = 4
    gpu_inference_streams = 1
    scenario = Scenario.Offline
    benchmark = Benchmark.SSDResNet34
    offline_expected_qps = 1000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_40GBx3(BenchmarkConfiguration):
    system = System("R7525_vA100-PCIE-40GB", Architecture.Ampere, 3)
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/coco/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/coco/val2017/SSDResNet34/int8_linear"
    use_graphs = False
    gpu_batch_size = 64
    gpu_copy_streams = 4
    gpu_inference_streams = 2
    run_infer_on_copy_streams = False
    scenario = Scenario.Offline
    benchmark = Benchmark.SSDResNet34
    offline_expected_qps = 2880
