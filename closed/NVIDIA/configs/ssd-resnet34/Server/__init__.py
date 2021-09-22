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
class A100_PCIex1(BenchmarkConfiguration):
    system = System("A100-PCIe", Architecture.Ampere, 1)
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
    server_target_qps = 770
    use_cuda_thread_per_device = True
    scenario = Scenario.Server
    benchmark = Benchmark.SSDResNet34


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIex1_Triton(BenchmarkConfiguration):
    system = System("A100-PCIe", Architecture.Ampere, 1)
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
    server_target_qps = 770
    use_cuda_thread_per_device = True
    scenario = Scenario.Server
    benchmark = Benchmark.SSDResNet34
    instance_group_count = 4
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIex8(BenchmarkConfiguration):
    system = System("A100-PCIe", Architecture.Ampere, 8)
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
    server_target_qps = 6160.0
    use_cuda_thread_per_device = True
    scenario = Scenario.Server
    benchmark = Benchmark.SSDResNet34


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIex8_Triton(BenchmarkConfiguration):
    system = System("A100-PCIe", Architecture.Ampere, 8)
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
    server_target_qps = 5800
    use_cuda_thread_per_device = True
    scenario = Scenario.Server
    benchmark = Benchmark.SSDResNet34
    instance_group_count = 4
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_PCIex8_MaxQ(BenchmarkConfiguration):
    system = System("A100-PCIe", Architecture.Ampere, 8)
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
    server_target_qps = 5700
    use_cuda_thread_per_device = True
    scenario = Scenario.Server
    benchmark = Benchmark.SSDResNet34
    power_limit = 200


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_PCIex8_Triton_MaxQ(BenchmarkConfiguration):
    system = System("A100-PCIe", Architecture.Ampere, 8)
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
    server_target_qps = 5600
    use_cuda_thread_per_device = True
    scenario = Scenario.Server
    benchmark = Benchmark.SSDResNet34
    instance_group_count = 4
    power_limit = 200
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx1(BenchmarkConfiguration):
    system = System("A100-PCIe-80GB", Architecture.Ampere, 1)
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
    server_target_qps = 770
    use_cuda_thread_per_device = True
    scenario = Scenario.Server
    benchmark = Benchmark.SSDResNet34


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx1_Triton(A100_PCIe_80GBx1):
    instance_group_count = 4
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx8(BenchmarkConfiguration):
    system = System("A100-PCIe-80GB", Architecture.Ampere, 8)
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
    server_target_qps = 6468.0
    use_cuda_thread_per_device = True
    scenario = Scenario.Server
    benchmark = Benchmark.SSDResNet34


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx8_Triton(A100_PCIe_80GBx8):
    server_target_qps = 5800
    instance_group_count = 4
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x1(BenchmarkConfiguration):
    system = System("A100-PCIe-80GB", Architecture.Ampere, 1, cpu_arch=CPUArch.aarch64)
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
    server_target_qps = 770
    use_cuda_thread_per_device = True
    scenario = Scenario.Server
    benchmark = Benchmark.SSDResNet34


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x2(BenchmarkConfiguration):
    system = System("A100-PCIe-80GB", Architecture.Ampere, 2, cpu_arch=CPUArch.aarch64)
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
    server_target_qps = 1540.0
    use_cuda_thread_per_device = True
    scenario = Scenario.Server
    benchmark = Benchmark.SSDResNet34


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x4(BenchmarkConfiguration):
    system = System("A100-PCIe-80GB", Architecture.Ampere, 4, cpu_arch=CPUArch.aarch64)
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
    server_target_qps = 3080.0
    use_cuda_thread_per_device = True
    scenario = Scenario.Server
    benchmark = Benchmark.SSDResNet34


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_PCIe_80GB_aarch64x4_MaxQ(A100_PCIe_80GB_aarch64x4):
    server_target_qps = 2750.0
    power_limit = 200


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_aarch64x1(BenchmarkConfiguration):
    system = System("A100-PCIe", Architecture.Ampere, 1, cpu_arch=CPUArch.aarch64)
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
    server_target_qps = 770
    use_cuda_thread_per_device = True
    scenario = Scenario.Server
    benchmark = Benchmark.SSDResNet34


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_aarch64x2(BenchmarkConfiguration):
    system = System("A100-PCIe", Architecture.Ampere, 2, cpu_arch=CPUArch.aarch64)
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
    server_target_qps = 1540.0
    use_cuda_thread_per_device = True
    scenario = Scenario.Server
    benchmark = Benchmark.SSDResNet34


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_aarch64x4(BenchmarkConfiguration):
    system = System("A100-PCIe", Architecture.Ampere, 4, cpu_arch=CPUArch.aarch64)
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
    server_target_qps = 3200.0
    use_cuda_thread_per_device = True
    scenario = Scenario.Server
    benchmark = Benchmark.SSDResNet34


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_PCIe_aarch64x4_MaxQ(A100_PCIe_aarch64x4):
    server_target_qps = 2850.0
    power_limit = 200


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb(BenchmarkConfiguration):
    _mig_configuration = MIGConfiguration({0: {MIGSlice(1, 10): 1}})
    system = System("A100-SXM-80GB", Architecture.Ampere, 1, mig_conf=_mig_configuration)
    active_sms = 100
    gpu_copy_streams = 2
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/coco/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/coco/val2017/SSDResNet34/int8_linear"
    use_deque_limit = True
    use_graphs = False
    deque_timeout_usec = 20000
    gpu_batch_size = 2
    gpu_inference_streams = 2
    server_target_qps = 100
    start_from_device = True
    use_cuda_thread_per_device = False
    scenario = Scenario.Server
    benchmark = Benchmark.SSDResNet34


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb_Hetero(A100_SXM_80GB_MIG_1x1g10gb):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb_Triton(BenchmarkConfiguration):
    _mig_configuration = MIGConfiguration({0: {MIGSlice(1, 10): 1}})
    system = System("A100-SXM-80GB", Architecture.Ampere, 1, mig_conf=_mig_configuration)
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
    server_target_qps = 100
    start_from_device = True
    use_cuda_thread_per_device = False
    scenario = Scenario.Server
    benchmark = Benchmark.SSDResNet34
    instance_group_count = 2
    use_triton = True


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
    server_target_qps = 7000
    start_from_device = True
    use_cuda_thread_per_device = False
    scenario = Scenario.Server
    benchmark = Benchmark.SSDResNet34


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
    server_target_qps = 5800
    start_from_device = True
    use_cuda_thread_per_device = False
    scenario = Scenario.Server
    benchmark = Benchmark.SSDResNet34
    instance_group_count = 2
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx1(BenchmarkConfiguration):
    system = System("A100-SXM-80GB", Architecture.Ampere, 1)
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
    server_target_qps = 925
    start_from_device = True
    use_cuda_thread_per_device = True
    scenario = Scenario.Server
    benchmark = Benchmark.SSDResNet34


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx1_Triton(BenchmarkConfiguration):
    system = System("A100-SXM-80GB", Architecture.Ampere, 1)
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
    server_target_qps = 910
    start_from_device = True
    use_cuda_thread_per_device = True
    scenario = Scenario.Server
    benchmark = Benchmark.SSDResNet34
    instance_group_count = 4
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx4(BenchmarkConfiguration):
    _system_alias = "DGX Station A100 - Red October"
    _notes = "This should not inherit from A100_SXM_80GB (DGX-A100), and cannot use start_from_device"

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
    gpu_inference_streams = 2
    server_target_qps = 3250
    use_cuda_thread_per_device = True
    scenario = Scenario.Server
    benchmark = Benchmark.SSDResNet34


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx4_Triton(BenchmarkConfiguration):
    _system_alias = "DGX Station A100 - Red October"
    _notes = "This should not inherit from A100_SXM_80GB (DGX-A100), and cannot use start_from_device"

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
    gpu_inference_streams = 2
    server_target_qps = 3144
    use_cuda_thread_per_device = True
    scenario = Scenario.Server
    benchmark = Benchmark.SSDResNet34
    instance_group_count = 4
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_SXM_80GBx4_MaxQ(BenchmarkConfiguration):
    _system_alias = "DGX Station A100 - Red October"
    _notes = "This should not inherit from A100_SXM_80GB (DGX-A100), and cannot use start_from_device"

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
    gpu_inference_streams = 2
    server_target_qps = 3080
    use_cuda_thread_per_device = True
    scenario = Scenario.Server
    benchmark = Benchmark.SSDResNet34
    power_limit = 250


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_SXM_80GBx4_Triton_MaxQ(BenchmarkConfiguration):
    _system_alias = "DGX Station A100 - Red October"
    _notes = "This should not inherit from A100_SXM_80GB (DGX-A100), and cannot use start_from_device"

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
    gpu_inference_streams = 2
    server_target_qps = 3144
    use_cuda_thread_per_device = True
    scenario = Scenario.Server
    benchmark = Benchmark.SSDResNet34
    instance_group_count = 4
    power_limit = 250
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
    server_target_qps = 7580
    start_from_device = True
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


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_SXM_80GBx8_MaxQ(BenchmarkConfiguration):
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
    server_target_qps = 6300
    start_from_device = True
    use_cuda_thread_per_device = True
    scenario = Scenario.Server
    benchmark = Benchmark.SSDResNet34
    power_limit = 250


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_SXM_80GBx8_Triton_MaxQ(BenchmarkConfiguration):
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
    power_limit = 250
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_40GBx1(BenchmarkConfiguration):
    system = System("A100-SXM4-40GB", Architecture.Ampere, 1)
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
    server_target_qps = 925
    start_from_device = True
    use_cuda_thread_per_device = True
    scenario = Scenario.Server
    benchmark = Benchmark.SSDResNet34


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_40GBx1_Triton(BenchmarkConfiguration):
    system = System("A100-SXM4-40GB", Architecture.Ampere, 1)
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
    server_target_qps = 910
    start_from_device = True
    use_cuda_thread_per_device = True
    scenario = Scenario.Server
    benchmark = Benchmark.SSDResNet34
    instance_group_count = 4
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_40GBx8(BenchmarkConfiguration):
    system = System("A100-SXM4-40GB", Architecture.Ampere, 8)
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
    server_target_qps = 7550
    start_from_device = True
    use_cuda_thread_per_device = True
    scenario = Scenario.Server
    benchmark = Benchmark.SSDResNet34


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_40GBx8_Triton(BenchmarkConfiguration):
    system = System("A100-SXM4-40GB", Architecture.Ampere, 8)
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


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A10x1(BenchmarkConfiguration):
    system = System("A10", Architecture.Ampere, 1)
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
    server_target_qps = 250
    use_cuda_thread_per_device = True
    scenario = Scenario.Server
    benchmark = Benchmark.SSDResNet34


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A10x1_Triton(BenchmarkConfiguration):
    system = System("A10", Architecture.Ampere, 1)
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
    server_target_qps = 230
    use_cuda_thread_per_device = True
    scenario = Scenario.Server
    benchmark = Benchmark.SSDResNet34
    instance_group_count = 4
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A10x8(BenchmarkConfiguration):
    system = System("A10", Architecture.Ampere, 8)
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
    server_target_qps = 2000.0
    use_cuda_thread_per_device = True
    scenario = Scenario.Server
    benchmark = Benchmark.SSDResNet34


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A10x8_Triton(BenchmarkConfiguration):
    system = System("A10", Architecture.Ampere, 8)
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
    server_target_qps = 2000
    use_cuda_thread_per_device = True
    scenario = Scenario.Server
    benchmark = Benchmark.SSDResNet34
    instance_group_count = 4
    use_triton = True
    max_queue_delay_usec = 500


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb(BenchmarkConfiguration):
    _mig_configuration = MIGConfiguration({0: {MIGSlice(1, 6): 1}})
    system = System("A30", Architecture.Ampere, 1, mig_conf=_mig_configuration)
    active_sms = 100
    gpu_copy_streams = 1
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/coco/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/coco/val2017/SSDResNet34/int8_linear"
    use_deque_limit = True
    use_graphs = False
    deque_timeout_usec = 50000
    gpu_batch_size = 2
    gpu_inference_streams = 1
    server_target_qps = 95
    use_cuda_thread_per_device = False
    workspace_size = 1610612736
    scenario = Scenario.Server
    benchmark = Benchmark.SSDResNet34


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_Hetero(A30_MIG_1x1g6gb):
    server_target_qps = 85


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_Triton(BenchmarkConfiguration):
    _mig_configuration = MIGConfiguration({0: {MIGSlice(1, 6): 1}})
    system = System("A30", Architecture.Ampere, 1, mig_conf=_mig_configuration)
    active_sms = 100
    gpu_copy_streams = 1
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/coco/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/coco/val2017/SSDResNet34/int8_linear"
    use_deque_limit = True
    use_graphs = False
    deque_timeout_usec = 50000
    gpu_batch_size = 2
    gpu_inference_streams = 1
    server_target_qps = 90
    use_cuda_thread_per_device = False
    workspace_size = 1610612736
    scenario = Scenario.Server
    benchmark = Benchmark.SSDResNet34
    instance_group_count = 4
    use_triton = True


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
    active_sms = 100
    gpu_copy_streams = 1
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/coco/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/coco/val2017/SSDResNet34/int8_linear"
    use_deque_limit = True
    use_graphs = False
    deque_timeout_usec = 50000
    gpu_batch_size = 2
    gpu_inference_streams = 1
    server_target_qps = 3000
    use_cuda_thread_per_device = False
    workspace_size = 1610612736
    scenario = Scenario.Server
    benchmark = Benchmark.SSDResNet34


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
    active_sms = 100
    gpu_copy_streams = 1
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/coco/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/coco/val2017/SSDResNet34/int8_linear"
    use_deque_limit = True
    use_graphs = False
    deque_timeout_usec = 50000
    gpu_batch_size = 2
    gpu_inference_streams = 1
    server_target_qps = 3000
    use_cuda_thread_per_device = False
    workspace_size = 1610612736
    scenario = Scenario.Server
    benchmark = Benchmark.SSDResNet34
    instance_group_count = 4
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x1(BenchmarkConfiguration):
    system = System("A30", Architecture.Ampere, 1)
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
    server_target_qps = 446.59999999999997
    use_cuda_thread_per_device = True
    scenario = Scenario.Server
    benchmark = Benchmark.SSDResNet34


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x1_Triton(BenchmarkConfiguration):
    system = System("A30", Architecture.Ampere, 1)
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
    server_target_qps = 446.59999999999997
    use_cuda_thread_per_device = True
    scenario = Scenario.Server
    benchmark = Benchmark.SSDResNet34
    instance_group_count = 4
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x8(BenchmarkConfiguration):
    system = System("A30", Architecture.Ampere, 8)
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
    server_target_qps = 3572.7999999999997
    use_cuda_thread_per_device = True
    scenario = Scenario.Server
    benchmark = Benchmark.SSDResNet34


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x8_Triton(BenchmarkConfiguration):
    system = System("A30", Architecture.Ampere, 8)
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
    instance_group_count = 4
    use_triton = True
    max_queue_delay_usec = 500


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A30x8_MaxQ(BenchmarkConfiguration):
    system = System("A30", Architecture.Ampere, 8)
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
    server_target_qps = 3305.9999999999995
    use_cuda_thread_per_device = True
    scenario = Scenario.Server
    benchmark = Benchmark.SSDResNet34
    power_limit = 200


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A30x8_Triton_MaxQ(BenchmarkConfiguration):
    system = System("A30", Architecture.Ampere, 8)
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
    server_target_qps = 3248.0
    use_cuda_thread_per_device = True
    scenario = Scenario.Server
    benchmark = Benchmark.SSDResNet34
    instance_group_count = 4
    power_limit = 200
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x1(BenchmarkConfiguration):
    system = System("T4", Architecture.Turing, 1)
    active_sms = 100
    gpu_copy_streams = 4
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/coco/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/coco/val2017/SSDResNet34/int8_linear"
    use_deque_limit = True
    use_graphs = False
    deque_timeout_usec = 2000
    gpu_batch_size = 2
    gpu_inference_streams = 4
    server_target_qps = 110
    use_cuda_thread_per_device = False
    scenario = Scenario.Server
    benchmark = Benchmark.SSDResNet34


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x1_Triton(BenchmarkConfiguration):
    system = System("T4", Architecture.Turing, 1)
    active_sms = 100
    gpu_copy_streams = 4
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/coco/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/coco/val2017/SSDResNet34/int8_linear"
    use_deque_limit = True
    use_graphs = False
    deque_timeout_usec = 2000
    gpu_batch_size = 2
    gpu_inference_streams = 4
    server_target_qps = 110
    use_cuda_thread_per_device = False
    scenario = Scenario.Server
    benchmark = Benchmark.SSDResNet34
    instance_group_count = 4
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x20(BenchmarkConfiguration):
    system = System("T4", Architecture.Turing, 20)
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
    gpu_batch_size = 4
    gpu_inference_streams = 2
    server_target_qps = 2400
    use_cuda_thread_per_device = True
    scenario = Scenario.Server
    benchmark = Benchmark.SSDResNet34


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x20_Triton(BenchmarkConfiguration):
    system = System("T4", Architecture.Turing, 20)
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
    gpu_batch_size = 2
    gpu_inference_streams = 2
    server_target_qps = 2280
    use_cuda_thread_per_device = True
    scenario = Scenario.Server
    benchmark = Benchmark.SSDResNet34
    instance_group_count = 2
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x8(BenchmarkConfiguration):
    system = System("T4", Architecture.Turing, 8)
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
    gpu_batch_size = 4
    gpu_inference_streams = 1
    server_target_qps = 1000
    use_cuda_thread_per_device = True
    scenario = Scenario.Server
    benchmark = Benchmark.SSDResNet34


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x8_Triton(BenchmarkConfiguration):
    system = System("T4", Architecture.Turing, 8)
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
    gpu_batch_size = 4
    gpu_inference_streams = 1
    server_target_qps = 720
    use_cuda_thread_per_device = True
    scenario = Scenario.Server
    benchmark = Benchmark.SSDResNet34
    instance_group_count = 2
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class Triton_CPU_2S_6258Rx1_Triton(BenchmarkConfiguration):
    system = System("Triton_CPU_2S_6258R", Architecture.Intel_CPU_x86_64, 1)
    active_sms = 100
    input_dtype = "fp32"
    map_path = "data_maps/coco/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/coco/val2017/SSDResNet34/fp32"
    use_deque_limit = True
    scenario = Scenario.Server
    benchmark = Benchmark.SSDResNet34
    batch_size = 1
    model_name = "ssd-resnet34_int8_openvino"
    num_instances = 4
    ov_parameters = {'CPU_THREADS_NUM': '56', 'CPU_THROUGHPUT_STREAMS': '2', 'ENABLE_BATCH_PADDING': 'NO', 'SKIP_OV_DYNAMIC_BATCHSIZE': 'YES'}
    server_target_qps = 22.5
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class Triton_CPU_4S_8380Hx1_Triton(BenchmarkConfiguration):
    system = System("Triton_CPU_4S_8380H", Architecture.Intel_CPU_x86_64, 1)
    active_sms = 100
    input_dtype = "fp32"
    map_path = "data_maps/coco/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/coco/val2017/SSDResNet34/fp32"
    use_deque_limit = True
    scenario = Scenario.Server
    benchmark = Benchmark.SSDResNet34
    batch_size = 1
    model_name = "ssd-resnet34_int8_openvino"
    num_instances = 8
    ov_parameters = {'CPU_THREADS_NUM': '112', 'CPU_THROUGHPUT_STREAMS': '4', 'ENABLE_BATCH_PADDING': 'NO', 'SKIP_OV_DYNAMIC_BATCHSIZE': 'YES'}
    request_timeout_usec = 64000
    server_target_qps = 79.5
    use_triton = True
