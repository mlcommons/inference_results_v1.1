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
class A100_PCIe_80GBx1(BenchmarkConfiguration):
    system = System("A100-PCIe-80GB", Architecture.Ampere, 1)
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
    offline_expected_qps = 960


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx1_Triton(BenchmarkConfiguration):
    system = System("A100-PCIe-80GB", Architecture.Ampere, 1)
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
    offline_expected_qps = 960


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx8(BenchmarkConfiguration):
    system = System("A100-PCIe-80GB", Architecture.Ampere, 8)
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
    offline_expected_qps = 7680.0


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx8_Triton(BenchmarkConfiguration):
    system = System("A100-PCIe-80GB", Architecture.Ampere, 8)
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
    offline_expected_qps = 7680.0


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_PCIe_80GBx8_MaxQ(BenchmarkConfiguration):
    system = System("A100-PCIe-80GB", Architecture.Ampere, 8)
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
    power_limit = 200
    offline_expected_qps = 5800


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_PCIe_80GBx8_Triton_MaxQ(BenchmarkConfiguration):
    system = System("A100-PCIe-80GB", Architecture.Ampere, 8)
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
    power_limit = 200
    use_triton = True
    offline_expected_qps = 7680.0


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x1(BenchmarkConfiguration):
    system = System("A100-PCIe-80GB", Architecture.Ampere, 1, cpu_arch=CPUArch.aarch64)
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
    offline_expected_qps = 960


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x2(BenchmarkConfiguration):
    system = System("A100-PCIe-80GB", Architecture.Ampere, 2, cpu_arch=CPUArch.aarch64)
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
    offline_expected_qps = 1920.0


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x4(BenchmarkConfiguration):
    system = System("A100-PCIe-80GB", Architecture.Ampere, 4, cpu_arch=CPUArch.aarch64)
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


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_PCIe_80GB_aarch64x4_MaxQ(A100_PCIe_80GB_aarch64x4):
    offline_expected_qps = 2900.0
    power_limit = 200


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_aarch64x1(BenchmarkConfiguration):
    system = System("A100-PCIe", Architecture.Ampere, 1, cpu_arch=CPUArch.aarch64)
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
    offline_expected_qps = 960


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_aarch64x2(BenchmarkConfiguration):
    system = System("A100-PCIe", Architecture.Ampere, 2, cpu_arch=CPUArch.aarch64)
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
    offline_expected_qps = 1920.0


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_aarch64x4(BenchmarkConfiguration):
    system = System("A100-PCIe", Architecture.Ampere, 4, cpu_arch=CPUArch.aarch64)
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


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_PCIe_aarch64x4_MaxQ(A100_PCIe_aarch64x4):
    offline_expected_qps = 2900.0
    power_limit = 200


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_MIG_1x1g5gb(BenchmarkConfiguration):
    _mig_configuration = MIGConfiguration({0: {MIGSlice(1, 5): 1}})
    system = System("A100-PCIe", Architecture.Ampere, 1, mig_conf=_mig_configuration)
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/coco/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/coco/val2017/SSDResNet34/int8_linear"
    use_graphs = False
    gpu_batch_size = 32
    gpu_copy_streams = 2
    gpu_inference_streams = 2
    scenario = Scenario.Offline
    benchmark = Benchmark.SSDResNet34
    offline_expected_qps = 135


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_MIG_1x1g5gb_Triton(BenchmarkConfiguration):
    _mig_configuration = MIGConfiguration({0: {MIGSlice(1, 5): 1}})
    system = System("A100-PCIe", Architecture.Ampere, 1, mig_conf=_mig_configuration)
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/coco/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/coco/val2017/SSDResNet34/int8_linear"
    use_graphs = False
    gpu_batch_size = 32
    gpu_copy_streams = 2
    gpu_inference_streams = 2
    scenario = Scenario.Offline
    benchmark = Benchmark.SSDResNet34
    use_triton = True
    offline_expected_qps = 135


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIex1(BenchmarkConfiguration):
    system = System("A100-PCIe", Architecture.Ampere, 1)
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
    offline_expected_qps = 960


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIex1_Triton(BenchmarkConfiguration):
    system = System("A100-PCIe", Architecture.Ampere, 1)
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
    offline_expected_qps = 960


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIex8(BenchmarkConfiguration):
    system = System("A100-PCIe", Architecture.Ampere, 8)
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
    offline_expected_qps = 7680.0


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIex8_Triton(BenchmarkConfiguration):
    system = System("A100-PCIe", Architecture.Ampere, 8)
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
    offline_expected_qps = 7680.0


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_PCIex8_MaxQ(BenchmarkConfiguration):
    system = System("A100-PCIe", Architecture.Ampere, 8)
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
    power_limit = 200
    offline_expected_qps = 5800


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_PCIex8_Triton_MaxQ(BenchmarkConfiguration):
    system = System("A100-PCIe", Architecture.Ampere, 8)
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
    power_limit = 200
    use_triton = True
    offline_expected_qps = 7680.0


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb(BenchmarkConfiguration):
    _mig_configuration = MIGConfiguration({0: {MIGSlice(1, 10): 1}})
    system = System("A100-SXM-80GB", Architecture.Ampere, 1, mig_conf=_mig_configuration)
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
    offline_expected_qps = 135


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb_Hetero(A100_SXM_80GB_MIG_1x1g10gb):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb_Triton(BenchmarkConfiguration):
    _mig_configuration = MIGConfiguration({0: {MIGSlice(1, 10): 1}})
    system = System("A100-SXM-80GB", Architecture.Ampere, 1, mig_conf=_mig_configuration)
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
    offline_expected_qps = 135


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_56x1g10gb(BenchmarkConfiguration):
    _mig_configuration = MIGConfiguration({
        0: {MIGSlice(1, 10): 7},
        1: {MIGSlice(1, 10): 7},
        2: {MIGSlice(1, 10): 7},
        3: {MIGSlice(1, 10): 7},
        4: {MIGSlice(1, 10): 7},
        5: {MIGSlice(1, 10): 7},
        6: {MIGSlice(1, 10): 7},
        7: {MIGSlice(1, 10): 7},
    })
    system = System("A100-SXM-80GB", Architecture.Ampere, 8, mig_conf=_mig_configuration)
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
    offline_expected_qps = 7560


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_56x1g10gb_Triton(BenchmarkConfiguration):
    _mig_configuration = MIGConfiguration({
        0: {MIGSlice(1, 10): 7},
        1: {MIGSlice(1, 10): 7},
        2: {MIGSlice(1, 10): 7},
        3: {MIGSlice(1, 10): 7},
        4: {MIGSlice(1, 10): 7},
        5: {MIGSlice(1, 10): 7},
        6: {MIGSlice(1, 10): 7},
        7: {MIGSlice(1, 10): 7},
    })
    system = System("A100-SXM-80GB", Architecture.Ampere, 8, mig_conf=_mig_configuration)
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
    offline_expected_qps = 7560


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx1(BenchmarkConfiguration):
    system = System("A100-SXM-80GB", Architecture.Ampere, 1)
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
    scenario = Scenario.Offline
    benchmark = Benchmark.SSDResNet34
    offline_expected_qps = 960


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx1_Triton(BenchmarkConfiguration):
    system = System("A100-SXM-80GB", Architecture.Ampere, 1)
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
    scenario = Scenario.Offline
    benchmark = Benchmark.SSDResNet34
    use_triton = True
    offline_expected_qps = 960


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx4(BenchmarkConfiguration):
    _system_alias = "DGX Station A100 - Red October"
    _notes = "This should not inherit from A100_SXM_80GB (DGX-A100), and cannot use start_from_device"

    system = System("A100-SXM-80GB", Architecture.Ampere, 4)
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
    offline_expected_qps = 3840


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx4_Triton(BenchmarkConfiguration):
    _system_alias = "DGX Station A100 - Red October"
    _notes = "This should not inherit from A100_SXM_80GB (DGX-A100), and cannot use start_from_device"

    system = System("A100-SXM-80GB", Architecture.Ampere, 4)
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
    offline_expected_qps = 3840


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_SXM_80GBx4_MaxQ(BenchmarkConfiguration):
    _system_alias = "DGX Station A100 - Red October"
    _notes = "This should not inherit from A100_SXM_80GB (DGX-A100), and cannot use start_from_device"

    system = System("A100-SXM-80GB", Architecture.Ampere, 4)
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
    power_limit = 250
    offline_expected_qps = 3840


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_SXM_80GBx4_Triton_MaxQ(BenchmarkConfiguration):
    _system_alias = "DGX Station A100 - Red October"
    _notes = "This should not inherit from A100_SXM_80GB (DGX-A100), and cannot use start_from_device"

    system = System("A100-SXM-80GB", Architecture.Ampere, 4)
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
    power_limit = 250
    use_triton = True
    offline_expected_qps = 3840


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx8(BenchmarkConfiguration):
    system = System("A100-SXM-80GB", Architecture.Ampere, 8)
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
    offline_expected_qps = 7800


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx8_Triton(BenchmarkConfiguration):
    system = System("A100-SXM-80GB", Architecture.Ampere, 8)
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
    offline_expected_qps = 7800


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_SXM_80GBx8_MaxQ(BenchmarkConfiguration):
    system = System("A100-SXM-80GB", Architecture.Ampere, 8)
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
    power_limit = 250
    offline_expected_qps = 7800


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_SXM_80GBx8_Triton_MaxQ(BenchmarkConfiguration):
    system = System("A100-SXM-80GB", Architecture.Ampere, 8)
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
    power_limit = 250
    use_triton = True
    offline_expected_qps = 7800


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_40GB_MIG_1x1g5gb(BenchmarkConfiguration):
    _mig_configuration = MIGConfiguration({0: {MIGSlice(1, 5): 1}})
    system = System("A100-SXM4-40GB", Architecture.Ampere, 1, mig_conf=_mig_configuration)
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
    offline_expected_qps = 135


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_40GB_MIG_1x1g5gb_Triton(BenchmarkConfiguration):
    _mig_configuration = MIGConfiguration({0: {MIGSlice(1, 5): 1}})
    system = System("A100-SXM4-40GB", Architecture.Ampere, 1, mig_conf=_mig_configuration)
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
    offline_expected_qps = 135


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_40GBx1(BenchmarkConfiguration):
    system = System("A100-SXM4-40GB", Architecture.Ampere, 1)
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
    scenario = Scenario.Offline
    benchmark = Benchmark.SSDResNet34
    offline_expected_qps = 960


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_40GBx1_Triton(BenchmarkConfiguration):
    system = System("A100-SXM4-40GB", Architecture.Ampere, 1)
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
    scenario = Scenario.Offline
    benchmark = Benchmark.SSDResNet34
    use_triton = True
    offline_expected_qps = 960


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_40GBx8(BenchmarkConfiguration):
    system = System("A100-SXM4-40GB", Architecture.Ampere, 8)
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/coco/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/coco/val2017/SSDResNet34/int8_linear"
    use_graphs = False
    gpu_batch_size = 64
    gpu_copy_streams = 4
    gpu_inference_streams = 1
    start_from_device = True
    scenario = Scenario.Offline
    benchmark = Benchmark.SSDResNet34
    offline_expected_qps = 7500


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_40GBx8_Triton(BenchmarkConfiguration):
    system = System("A100-SXM4-40GB", Architecture.Ampere, 8)
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/coco/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/coco/val2017/SSDResNet34/int8_linear"
    use_graphs = False
    gpu_batch_size = 64
    gpu_copy_streams = 4
    gpu_inference_streams = 1
    scenario = Scenario.Offline
    benchmark = Benchmark.SSDResNet34
    use_triton = True
    offline_expected_qps = 7500


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A10x1(BenchmarkConfiguration):
    system = System("A10", Architecture.Ampere, 1)
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/coco/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/coco/val2017/SSDResNet34/int8_linear"
    use_graphs = False
    gpu_batch_size = 32
    gpu_copy_streams = 2
    gpu_inference_streams = 1
    scenario = Scenario.Offline
    benchmark = Benchmark.SSDResNet34
    offline_expected_qps = 310


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A10x1_Triton(BenchmarkConfiguration):
    system = System("A10", Architecture.Ampere, 1)
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/coco/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/coco/val2017/SSDResNet34/int8_linear"
    use_graphs = False
    gpu_batch_size = 32
    gpu_copy_streams = 2
    gpu_inference_streams = 1
    scenario = Scenario.Offline
    benchmark = Benchmark.SSDResNet34
    use_triton = True
    offline_expected_qps = 310


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A10x8(BenchmarkConfiguration):
    system = System("A10", Architecture.Ampere, 8)
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/coco/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/coco/val2017/SSDResNet34/int8_linear"
    use_graphs = False
    gpu_batch_size = 32
    gpu_copy_streams = 2
    gpu_inference_streams = 1
    scenario = Scenario.Offline
    benchmark = Benchmark.SSDResNet34
    offline_expected_qps = 2480.0


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A10x8_Triton(BenchmarkConfiguration):
    system = System("A10", Architecture.Ampere, 8)
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/coco/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/coco/val2017/SSDResNet34/int8_linear"
    use_graphs = False
    gpu_batch_size = 32
    gpu_copy_streams = 2
    gpu_inference_streams = 1
    scenario = Scenario.Offline
    benchmark = Benchmark.SSDResNet34
    use_triton = True
    offline_expected_qps = 2470


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb(BenchmarkConfiguration):
    _mig_configuration = MIGConfiguration({0: {MIGSlice(1, 6): 1}})
    system = System("A30", Architecture.Ampere, 1, mig_conf=_mig_configuration)
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/coco/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/coco/val2017/SSDResNet34/int8_linear"
    use_graphs = False
    gpu_batch_size = 4
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    workspace_size = 1610612736
    scenario = Scenario.Offline
    benchmark = Benchmark.SSDResNet34
    offline_expected_qps = 128


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_Hetero(A30_MIG_1x1g6gb):
    offline_expected_qps = 115


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_Triton(BenchmarkConfiguration):
    _mig_configuration = MIGConfiguration({0: {MIGSlice(1, 6): 1}})
    system = System("A30", Architecture.Ampere, 1, mig_conf=_mig_configuration)
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/coco/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/coco/val2017/SSDResNet34/int8_linear"
    use_graphs = False
    gpu_batch_size = 4
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    workspace_size = 1610612736
    scenario = Scenario.Offline
    benchmark = Benchmark.SSDResNet34
    use_triton = True
    offline_expected_qps = 130


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_32x1g6gb(BenchmarkConfiguration):
    _mig_configuration = MIGConfiguration({
        0: {MIGSlice(1, 6): 4},
        1: {MIGSlice(1, 6): 4},
        2: {MIGSlice(1, 6): 4},
        3: {MIGSlice(1, 6): 4},
        4: {MIGSlice(1, 6): 4},
        5: {MIGSlice(1, 6): 4},
        6: {MIGSlice(1, 6): 4},
        7: {MIGSlice(1, 6): 4},
    })
    system = System("A30", Architecture.Ampere, 8, mig_conf=_mig_configuration)
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/coco/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/coco/val2017/SSDResNet34/int8_linear"
    use_graphs = False
    gpu_batch_size = 4
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    workspace_size = 1610612736
    scenario = Scenario.Offline
    benchmark = Benchmark.SSDResNet34
    offline_expected_qps = 3840


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_32x1g6gb_Triton(BenchmarkConfiguration):
    _mig_configuration = MIGConfiguration({
        0: {MIGSlice(1, 6): 4},
        1: {MIGSlice(1, 6): 4},
        2: {MIGSlice(1, 6): 4},
        3: {MIGSlice(1, 6): 4},
        4: {MIGSlice(1, 6): 4},
        5: {MIGSlice(1, 6): 4},
        6: {MIGSlice(1, 6): 4},
        7: {MIGSlice(1, 6): 4},
    })
    system = System("A30", Architecture.Ampere, 8, mig_conf=_mig_configuration)
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/coco/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/coco/val2017/SSDResNet34/int8_linear"
    use_graphs = False
    gpu_batch_size = 4
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    workspace_size = 1610612736
    scenario = Scenario.Offline
    benchmark = Benchmark.SSDResNet34
    use_triton = True
    offline_expected_qps = 4160


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x1(BenchmarkConfiguration):
    system = System("A30", Architecture.Ampere, 1)
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
    offline_expected_qps = 470


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x1_Triton(BenchmarkConfiguration):
    system = System("A30", Architecture.Ampere, 1)
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
    use_triton = True
    offline_expected_qps = 470


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x8(BenchmarkConfiguration):
    system = System("A30", Architecture.Ampere, 8)
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


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x8_Triton(BenchmarkConfiguration):
    system = System("A30", Architecture.Ampere, 8)
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
    use_triton = True
    offline_expected_qps = 3760.0


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class AGX_Xavier(BenchmarkConfiguration):
    system = System("AGX_Xavier", Architecture.Xavier, 1, cpu_arch=CPUArch.aarch64)
    # GPU-only QPS
    _gpu_offline_expected_qps = 35.1243
    # DLA-only (1x) QPS
    _dla_offline_expected_qps = 10
    # GPU + 2 DLA QPS
    offline_expected_qps = 56

    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/coco/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/coco/val2017/SSDResNet34/int8_linear"
    use_graphs = False
    dla_batch_size = 1
    dla_copy_streams = 1
    dla_core = 0
    dla_inference_streams = 1
    gpu_batch_size = 2
    gpu_copy_streams = 4
    gpu_inference_streams = 1
    scenario = Scenario.Offline
    benchmark = Benchmark.SSDResNet34


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class AGX_Xavier_MaxQ(BenchmarkConfiguration):
    system = System("AGX_Xavier", Architecture.Xavier, 1, cpu_arch=CPUArch.aarch64)
    # GPU-only QPS
    _gpu_offline_expected_qps = 35.1243
    # DLA-only (1x) QPS
    _dla_offline_expected_qps = 10
    # GPU + 2 DLA QPS
    offline_expected_qps = 42

    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/coco/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/coco/val2017/SSDResNet34/int8_linear"
    use_graphs = False
    dla_batch_size = 1
    dla_copy_streams = 1
    dla_core = 0
    dla_inference_streams = 1
    gpu_batch_size = 2
    gpu_copy_streams = 4
    gpu_inference_streams = 1
    scenario = Scenario.Offline
    benchmark = Benchmark.SSDResNet34

    # power settings
    xavier_gpu_freq = 905250000
    xavier_dla_freq = 950000000
    xavier_cpu_freq = 1190400
    xavier_emc_freq = 1331200000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class AGX_Xavier_Triton(BenchmarkConfiguration):
    system = System("AGX_Xavier", Architecture.Xavier, 1, cpu_arch=CPUArch.aarch64)
    # GPU-only QPS
    _gpu_offline_expected_qps = 35.1243
    # DLA-only (1x) QPS
    _dla_offline_expected_qps = 10
    # GPU + 2 DLA QPS
    offline_expected_qps = 56

    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/coco/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/coco/val2017/SSDResNet34/int8_linear"
    use_graphs = False
    dla_batch_size = 1
    dla_copy_streams = 1
    dla_core = 0
    dla_inference_streams = 1
    gpu_batch_size = 2
    gpu_copy_streams = 4
    gpu_inference_streams = 1
    scenario = Scenario.Offline
    benchmark = Benchmark.SSDResNet34
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x1(BenchmarkConfiguration):
    system = System("T4", Architecture.Turing, 1)
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/coco/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/coco/val2017/SSDResNet34/int8_linear"
    use_graphs = False
    gpu_batch_size = 12
    gpu_copy_streams = 4
    gpu_inference_streams = 1
    scenario = Scenario.Offline
    benchmark = Benchmark.SSDResNet34
    offline_expected_qps = 140


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x1_Triton(BenchmarkConfiguration):
    system = System("T4", Architecture.Turing, 1)
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/coco/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/coco/val2017/SSDResNet34/int8_linear"
    use_graphs = False
    gpu_batch_size = 12
    gpu_copy_streams = 4
    gpu_inference_streams = 1
    scenario = Scenario.Offline
    benchmark = Benchmark.SSDResNet34
    use_triton = True
    offline_expected_qps = 140


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x20(BenchmarkConfiguration):
    system = System("T4", Architecture.Turing, 20)
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/coco/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/coco/val2017/SSDResNet34/int8_linear"
    use_graphs = False
    gpu_batch_size = 12
    gpu_copy_streams = 4
    gpu_inference_streams = 1
    scenario = Scenario.Offline
    benchmark = Benchmark.SSDResNet34
    offline_expected_qps = 2800


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x20_Triton(BenchmarkConfiguration):
    system = System("T4", Architecture.Turing, 20)
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/coco/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/coco/val2017/SSDResNet34/int8_linear"
    use_graphs = False
    gpu_batch_size = 12
    gpu_copy_streams = 4
    gpu_inference_streams = 1
    scenario = Scenario.Offline
    benchmark = Benchmark.SSDResNet34
    use_triton = True
    offline_expected_qps = 2800


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x8(BenchmarkConfiguration):
    system = System("T4", Architecture.Turing, 8)
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/coco/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/coco/val2017/SSDResNet34/int8_linear"
    use_graphs = False
    gpu_batch_size = 12
    gpu_copy_streams = 4
    gpu_inference_streams = 1
    scenario = Scenario.Offline
    benchmark = Benchmark.SSDResNet34
    offline_expected_qps = 1116


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x8_Triton(BenchmarkConfiguration):
    system = System("T4", Architecture.Turing, 8)
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/coco/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/coco/val2017/SSDResNet34/int8_linear"
    use_graphs = False
    gpu_batch_size = 12
    gpu_copy_streams = 4
    gpu_inference_streams = 1
    scenario = Scenario.Offline
    benchmark = Benchmark.SSDResNet34
    use_triton = True
    offline_expected_qps = 1116


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class Triton_CPU_2S_6258Rx1_Triton(BenchmarkConfiguration):
    system = System("Triton_CPU_2S_6258R", Architecture.Intel_CPU_x86_64, 1)
    input_dtype = "fp32"
    map_path = "data_maps/coco/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/coco/val2017/SSDResNet34/fp32"
    batch_size = 1
    offline_expected_qps = 48
    run_infer_on_copy_streams = False
    scenario = Scenario.Offline
    benchmark = Benchmark.SSDResNet34
    max_queue_delay_usec = 100
    model_name = "ssd-resnet34_int8_openvino"
    num_instances = 56
    ov_parameters = {'CPU_THREADS_NUM': '56', 'CPU_THROUGHPUT_STREAMS': '56', 'ENABLE_BATCH_PADDING': 'NO', 'SKIP_OV_DYNAMIC_BATCHSIZE': 'YES'}
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class Triton_CPU_4S_8380Hx1_Triton(BenchmarkConfiguration):
    system = System("Triton_CPU_4S_8380H", Architecture.Intel_CPU_x86_64, 1)
    input_dtype = "fp32"
    map_path = "data_maps/coco/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/coco/val2017/SSDResNet34/fp32"
    batch_size = 1
    offline_expected_qps = 117
    run_infer_on_copy_streams = False
    scenario = Scenario.Offline
    benchmark = Benchmark.SSDResNet34
    max_queue_delay_usec = 100
    model_name = "ssd-resnet34_int8_openvino"
    num_instances = 112
    ov_parameters = {'CPU_THREADS_NUM': '112', 'CPU_THROUGHPUT_STREAMS': '112', 'ENABLE_BATCH_PADDING': 'NO', 'SKIP_OV_DYNAMIC_BATCHSIZE': 'YES'}
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class Xavier_NX(BenchmarkConfiguration):
    system = System("Xavier_NX", Architecture.Xavier, 1, cpu_arch=CPUArch.aarch64)
    # GPU-only QPS
    _gpu_offline_expected_qps = 17
    # DLA-only (1x) QPS
    _dla_offline_expected_qps = 10
    # GPU + 2 DLA QPS
    offline_expected_qps = 40

    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/coco/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/coco/val2017/SSDResNet34/int8_linear"
    use_graphs = False
    dla_batch_size = 1
    dla_copy_streams = 1
    dla_core = 0
    dla_inference_streams = 1
    gpu_batch_size = 1
    gpu_copy_streams = 4
    gpu_inference_streams = 1
    scenario = Scenario.Offline
    benchmark = Benchmark.SSDResNet34


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class Xavier_NX_MaxQ(BenchmarkConfiguration):
    system = System("Xavier_NX", Architecture.Xavier, 1, cpu_arch=CPUArch.aarch64)
    # GPU-only QPS
    _gpu_offline_expected_qps = 17
    # DLA-only (1x) QPS
    _dla_offline_expected_qps = 10
    # GPU + 2 DLA QPS
    offline_expected_qps = 40

    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/coco/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/coco/val2017/SSDResNet34/int8_linear"
    use_graphs = False
    dla_batch_size = 1
    dla_copy_streams = 1
    dla_core = 0
    dla_inference_streams = 1
    gpu_batch_size = 1
    gpu_copy_streams = 4
    gpu_inference_streams = 1
    scenario = Scenario.Offline
    benchmark = Benchmark.SSDResNet34

    # power settings
    xavier_gpu_freq = 803250000
    xavier_dla_freq = 1100800000
    xavier_cpu_freq = 1190400
    xavier_emc_freq = 1331200000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class Xavier_NX_Triton(BenchmarkConfiguration):
    system = System("Xavier_NX", Architecture.Xavier, 1, cpu_arch=CPUArch.aarch64)
    # GPU-only QPS
    _gpu_offline_expected_qps = 17
    # DLA-only (1x) QPS
    _dla_offline_expected_qps = 10
    # GPU + 2 DLA QPS
    offline_expected_qps = 40

    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/coco/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/coco/val2017/SSDResNet34/int8_linear"
    use_graphs = False
    dla_batch_size = 1
    dla_copy_streams = 1
    dla_core = 0
    dla_inference_streams = 1
    gpu_batch_size = 1
    gpu_copy_streams = 4
    gpu_inference_streams = 1
    scenario = Scenario.Offline
    benchmark = Benchmark.SSDResNet34
    use_triton = True
