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
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/imagenet/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/imagenet/ResNet50/int8_linear"
    use_graphs = False
    gpu_batch_size = 2048
    gpu_copy_streams = 2
    run_infer_on_copy_streams = False
    scenario = Scenario.Offline
    benchmark = Benchmark.ResNet50
    offline_expected_qps = 330000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class DSS8440_A30x8(BenchmarkConfiguration):
    system = System("DSS8440_A30", Architecture.Ampere, 8)
    gpu_inference_streams = 4
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/imagenet/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/imagenet/ResNet50/int8_linear"
    use_graphs = False
    gpu_batch_size = 2048
    gpu_copy_streams = 4
    run_infer_on_copy_streams = True
    scenario = Scenario.Offline
    benchmark = Benchmark.ResNet50
    offline_expected_qps = 168000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class DSS8440_A30x8_Triton(BenchmarkConfiguration):
    system = System("DSS8440_A30", Architecture.Ampere, 8)
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/imagenet/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/imagenet/ResNet50/int8_linear"
    use_graphs = False
    gpu_batch_size = 2048
    gpu_copy_streams = 5
    run_infer_on_copy_streams = True
    scenario = Scenario.Offline
    benchmark = Benchmark.ResNet50
    use_triton = True
    offline_expected_qps = 151200.0


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class R750xa_A100_PCIE_40GBx4_MaxQ(BenchmarkConfiguration):
    system = System("R750xa_A100-PCIE-40GB", Architecture.Ampere, 4)
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/imagenet/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/imagenet/ResNet50/int8_linear"
    use_graphs = False
    gpu_batch_size = 2048
    gpu_copy_streams = 2
    run_infer_on_copy_streams = False
    scenario = Scenario.Offline
    benchmark = Benchmark.ResNet50
    offline_expected_qps = 32000*4
    power_limit = 175
    start_from_device= True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750xa_A100_PCIE_80GBx4(BenchmarkConfiguration):
    system = System("R750xa_A100-PCIE-80GB", Architecture.Ampere, 4)
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/imagenet/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/imagenet/ResNet50/int8_linear"
    use_graphs = False
    gpu_batch_size = 2048
    gpu_copy_streams = 2
    run_infer_on_copy_streams = False
    scenario = Scenario.Offline
    benchmark = Benchmark.ResNet50
    offline_expected_qps = 147264


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750xa_A100_PCIE_80GBx4_Triton(BenchmarkConfiguration):
    system = System("R750xa_A100-PCIE-80GB", Architecture.Ampere, 4)
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/imagenet/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/imagenet/ResNet50/int8_linear"
    use_graphs = False
    gpu_batch_size = 2048
    gpu_copy_streams = 2
    run_infer_on_copy_streams = False
    scenario = Scenario.Offline
    benchmark = Benchmark.ResNet50
    use_triton = True
    offline_expected_qps = 147264

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750xa_A100_PCIE_80GB_MIG_28x1g10gb(BenchmarkConfiguration):
    _mig_configuration = MIGConfiguration({
        0: {MIGSlice(1, 10): 7},
        1: {MIGSlice(1, 10): 7},
        2: {MIGSlice(1, 10): 7},
        3: {MIGSlice(1, 10): 7},
    })
    system = System("R750xa_A100-PCIE-80GB", Architecture.Ampere, 4, mig_conf=_mig_configuration)
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/imagenet/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/imagenet/ResNet50/int8_linear"
    use_graphs = False
    gpu_batch_size = 256
    gpu_copy_streams = 2
    run_infer_on_copy_streams = True
    start_from_device = True
    scenario = Scenario.Offline
    benchmark = Benchmark.ResNet50
    offline_expected_qps = 155600


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750xa_A100_PCIE_80GB_MIG_28x1g10gb_Triton(BenchmarkConfiguration):
    _mig_configuration = MIGConfiguration({
        0: {MIGSlice(1, 10): 7},
        1: {MIGSlice(1, 10): 7},
        2: {MIGSlice(1, 10): 7},
        3: {MIGSlice(1, 10): 7},
    })
    system = System("R750xa_A100-PCIE-80GB", Architecture.Ampere, 4, mig_conf=_mig_configuration)
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/imagenet/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/imagenet/ResNet50/int8_linear"
    use_graphs = False
    gpu_batch_size = 256
    gpu_copy_streams = 2
    run_infer_on_copy_streams = True
    start_from_device = True
    scenario = Scenario.Offline
    benchmark = Benchmark.ResNet50
    use_triton = True
    offline_expected_qps = 155600


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R7525_A100_PCIE_40GBx3(BenchmarkConfiguration):
    system = System("R7525_A100-PCIE-40GB", Architecture.Ampere, 3)
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/imagenet/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/imagenet/ResNet50/int8_linear"
    use_graphs = False
    gpu_batch_size = 2048
    gpu_copy_streams = 2
    run_infer_on_copy_streams = False
    scenario = Scenario.Offline
    benchmark = Benchmark.ResNet50
    offline_expected_qps = 32000*3
    start_from_device=True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R7525_A30x3(BenchmarkConfiguration):
    system = System("R7525_A30", Architecture.Ampere, 3)
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/imagenet/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/imagenet/ResNet50/int8_linear"
    use_graphs = False
    gpu_batch_size = 1536
    gpu_copy_streams = 2
    run_infer_on_copy_streams = True
    scenario = Scenario.Offline
    benchmark = Benchmark.ResNet50
    offline_expected_qps = 18200*3


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE2420_A10x1(BenchmarkConfiguration):
    system = System("XE2420_A10", Architecture.Ampere, 1)
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/imagenet/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/imagenet/ResNet50/int8_linear"
    use_graphs = False
    gpu_batch_size = 1024
    gpu_copy_streams = 2
    scenario = Scenario.Offline
    benchmark = Benchmark.ResNet50
    offline_expected_qps = 18000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class XE2420_A10x1_MaxQ(XE2420_A10x1):
    pass


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE2420_A10x2(XE2420_A10x1):
    system = System("XE2420_A10", Architecture.Ampere, 2)
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/imagenet/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/imagenet/ResNet50/int8_linear"
    use_graphs = False
    gpu_batch_size = 1024
    gpu_copy_streams = 2
    scenario = Scenario.Offline
    benchmark = Benchmark.ResNet50
    offline_expected_qps = 30000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class XE2420_A10x2_MaxQ(XE2420_A10x2):
    pass


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8545_A100_SXM_80GBx4(BenchmarkConfiguration):
    system = System("XE8545_A100-SXM-80GB", Architecture.Ampere, 4)
    gpu_inference_streams = 2
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/imagenet/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/imagenet/ResNet50/int8_linear"
    use_graphs = False
    gpu_batch_size = 2048
    gpu_copy_streams = 3
    run_infer_on_copy_streams = False
    start_from_device = True
    scenario = Scenario.Offline
    benchmark = Benchmark.ResNet50
    offline_expected_qps = 168400


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8545_A100_SXM_80GBx4_Triton(BenchmarkConfiguration):
    system = System("XE8545_A100-SXM-80GB", Architecture.Ampere, 4)
    gpu_inference_streams = 2
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/imagenet/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/imagenet/ResNet50/int8_linear"
    use_graphs = False
    gpu_batch_size = 2048
    gpu_copy_streams = 1
    run_infer_on_copy_streams = False
    scenario = Scenario.Offline
    benchmark = Benchmark.ResNet50
    use_triton = True
    offline_expected_qps = 160664
    start_from_device = False


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8545_A100_SXM_80GB_MIG_28x1g10gb_Triton(BenchmarkConfiguration):
    _mig_configuration = MIGConfiguration({
        0: {MIGSlice(1, 10): 7},
        1: {MIGSlice(1, 10): 7},
        2: {MIGSlice(1, 10): 7},
        3: {MIGSlice(1, 10): 7},
    })
    system = System("XE8545_A100-SXM-80GB", Architecture.Ampere, 4, mig_conf=_mig_configuration)
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/imagenet/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/imagenet/ResNet50/int8_linear"
    use_graphs = False
    gpu_batch_size = 256
    gpu_copy_streams = 2
    run_infer_on_copy_streams = True
    start_from_device = False
    scenario = Scenario.Offline
    benchmark = Benchmark.ResNet50
    use_triton = True
    offline_expected_qps = 145420


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR12_A10x2(BenchmarkConfiguration):
    system = System("XR12_A10", Architecture.Ampere, 2)
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/imagenet/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/imagenet/ResNet50/int8_linear"
    use_graphs = False
    gpu_batch_size = 1024
    gpu_copy_streams = 2
    scenario = Scenario.Offline
    benchmark = Benchmark.ResNet50
    offline_expected_qps = 30000

