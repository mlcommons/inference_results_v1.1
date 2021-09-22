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
    server_target_qps = 26000
    use_cuda_thread_per_device = True
    use_graphs = True
    scenario = Scenario.Server
    benchmark = Benchmark.ResNet50


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIex1_Triton(BenchmarkConfiguration):
    system = System("A100-PCIe", Architecture.Ampere, 1)
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
    server_target_qps = 24000
    use_cuda_thread_per_device = True
    use_graphs = False
    scenario = Scenario.Server
    benchmark = Benchmark.ResNet50
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIex8(BenchmarkConfiguration):
    system = System("A100-PCIe", Architecture.Ampere, 8)
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
    gpu_inference_streams = 5
    server_target_qps = 190000
    numa_config = "3:0-15&2:16-31&1:32-47&0:48-63&7:64-79&6:80-95&5:96-111&4:112-127"
    use_cuda_thread_per_device = True
    use_graphs = True
    scenario = Scenario.Server
    benchmark = Benchmark.ResNet50


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIex8_Triton(BenchmarkConfiguration):
    system = System("A100-PCIe", Architecture.Ampere, 8)
    active_sms = 100
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/imagenet/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/imagenet/ResNet50/int8_linear"
    use_deque_limit = True
    deque_timeout_usec = 500
    gpu_batch_size = 64
    gpu_copy_streams = 4
    gpu_inference_streams = 3
    server_target_qps = 175000
    use_cuda_thread_per_device = True
    numa_config = "3:0-7&2:8-15&1:16-23&0:24-31&7:32-39&6:40-47&5:48-55&4:56-63"
    use_graphs = False
    scenario = Scenario.Server
    benchmark = Benchmark.ResNet50
    use_triton = True
    max_queue_delay_usec = 1000
    request_timeout_usec = 2000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_PCIex8_MaxQ(BenchmarkConfiguration):
    system = System("A100-PCIe", Architecture.Ampere, 8)
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
    server_target_qps = 185000
    use_cuda_thread_per_device = True
    use_graphs = True
    scenario = Scenario.Server
    benchmark = Benchmark.ResNet50
    power_limit = 200


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_PCIex8_Triton_MaxQ(BenchmarkConfiguration):
    system = System("A100-PCIe", Architecture.Ampere, 8)
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
    server_target_qps = 130000
    use_cuda_thread_per_device = True
    use_graphs = False
    scenario = Scenario.Server
    benchmark = Benchmark.ResNet50
    power_limit = 175
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx1(BenchmarkConfiguration):
    system = System("A100-PCIe-80GB", Architecture.Ampere, 1)
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
    server_target_qps = 26000
    use_cuda_thread_per_device = True
    use_graphs = True
    scenario = Scenario.Server
    benchmark = Benchmark.ResNet50


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx1_Triton(A100_PCIe_80GBx1):
    server_target_qps = 24000
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx8(BenchmarkConfiguration):
    system = System("A100-PCIe-80GB", Architecture.Ampere, 8)
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
    gpu_inference_streams = 5
    server_target_qps = 200000
    use_cuda_thread_per_device = True
    use_graphs = True
    scenario = Scenario.Server
    benchmark = Benchmark.ResNet50


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx8_Triton(BenchmarkConfiguration):
    system = System("A100-PCIe-80GB", Architecture.Ampere, 8)
    active_sms = 100
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/imagenet/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/imagenet/ResNet50/int8_linear"
    use_deque_limit = True
    deque_timeout_usec = 500
    gpu_batch_size = 64
    gpu_copy_streams = 4
    gpu_inference_streams = 3
    server_target_qps = 181000
    use_cuda_thread_per_device = True
    numa_config = "3:0-15&2:16-31&1:32-47&0:48-63&7:64-79&6:80-95&5:96-111&4:112-127"
    use_graphs = False
    scenario = Scenario.Server
    benchmark = Benchmark.ResNet50
    use_triton = True
    max_queue_delay_usec = 1000
    request_timeout_usec = 2000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x1(BenchmarkConfiguration):
    system = System("A100-PCIe-80GB", Architecture.Ampere, 1, cpu_arch=CPUArch.aarch64)
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
    server_target_qps = 26000
    use_cuda_thread_per_device = True
    use_graphs = True
    scenario = Scenario.Server
    benchmark = Benchmark.ResNet50


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x2(BenchmarkConfiguration):
    system = System("A100-PCIe-80GB", Architecture.Ampere, 2, cpu_arch=CPUArch.aarch64)
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
    gpu_inference_streams = 5
    server_target_qps = 52000
    use_cuda_thread_per_device = True
    use_graphs = True
    scenario = Scenario.Server
    benchmark = Benchmark.ResNet50


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x4(BenchmarkConfiguration):
    system = System("A100-PCIe-80GB", Architecture.Ampere, 4, cpu_arch=CPUArch.aarch64)
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
    gpu_inference_streams = 5
    server_target_qps = 104000
    use_cuda_thread_per_device = True
    use_graphs = True
    scenario = Scenario.Server
    benchmark = Benchmark.ResNet50


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_PCIe_80GB_aarch64x4_MaxQ(A100_PCIe_80GB_aarch64x4):
    server_target_qps = 92500
    power_limit = 175


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_aarch64x1(BenchmarkConfiguration):
    system = System("A100-PCIe", Architecture.Ampere, 1, cpu_arch=CPUArch.aarch64)
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
    server_target_qps = 26000
    use_cuda_thread_per_device = True
    use_graphs = True
    scenario = Scenario.Server
    benchmark = Benchmark.ResNet50


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_aarch64x2(BenchmarkConfiguration):
    system = System("A100-PCIe", Architecture.Ampere, 2, cpu_arch=CPUArch.aarch64)
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
    gpu_inference_streams = 5
    server_target_qps = 52000
    use_cuda_thread_per_device = True
    use_graphs = True
    scenario = Scenario.Server
    benchmark = Benchmark.ResNet50


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_aarch64x4(BenchmarkConfiguration):
    system = System("A100-PCIe", Architecture.Ampere, 4, cpu_arch=CPUArch.aarch64)
    active_sms = 100
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/imagenet/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/imagenet/ResNet50/int8_linear"
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 256
    gpu_copy_streams = 2
    gpu_inference_streams = 1
    server_target_qps = 115000
    use_cuda_thread_per_device = True
    use_graphs = True
    scenario = Scenario.Server
    benchmark = Benchmark.ResNet50


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_PCIe_aarch64x4_MaxQ(A100_PCIe_aarch64x4):
    gpu_batch_size = 256
    gpu_copy_streams = 2
    gpu_inference_streams = 1
    server_target_qps = 101000
    power_limit = 175


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb(BenchmarkConfiguration):
    _mig_configuration = MIGConfiguration({0: {MIGSlice(1, 10): 1}})
    system = System("A100-SXM-80GB", Architecture.Ampere, 1, mig_conf=_mig_configuration)
    active_sms = 100
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/imagenet/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/imagenet/ResNet50/int8_linear"
    use_deque_limit = True
    deque_timeout_usec = 1000
    gpu_batch_size = 8
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    server_target_qps = 3600
    start_from_device = True
    use_graphs = True
    scenario = Scenario.Server
    benchmark = Benchmark.ResNet50


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb_Hetero(A100_SXM_80GB_MIG_1x1g10gb):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb_Triton(BenchmarkConfiguration):
    _mig_configuration = MIGConfiguration({0: {MIGSlice(1, 10): 1}})
    system = System("A100-SXM-80GB", Architecture.Ampere, 1, mig_conf=_mig_configuration)
    active_sms = 100
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/imagenet/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/imagenet/ResNet50/int8_linear"
    use_deque_limit = True
    deque_timeout_usec = 1000
    gpu_batch_size = 8
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    server_target_qps = 3500
    start_from_device = True
    use_graphs = False
    scenario = Scenario.Server
    benchmark = Benchmark.ResNet50
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
    server_target_qps = 220000
    start_from_device = True
    use_graphs = True
    scenario = Scenario.Server
    benchmark = Benchmark.ResNet50


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
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/imagenet/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/imagenet/ResNet50/int8_linear"
    use_deque_limit = True
    deque_timeout_usec = 10
    gpu_batch_size = 4
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    server_target_qps = 70000
    use_graphs = False
    start_from_device = True
    scenario = Scenario.Server
    benchmark = Benchmark.ResNet50
    use_triton = True
    instance_group_count = 1
    max_queue_delay_usec = 10
    preferred_batch_size = 1


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx1(BenchmarkConfiguration):
    system = System("A100-SXM-80GB", Architecture.Ampere, 1)
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
    gpu_inference_streams = 4
    server_target_qps = 30800
    start_from_device = True
    use_cuda_thread_per_device = True
    use_graphs = True
    scenario = Scenario.Server
    benchmark = Benchmark.ResNet50


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx1_Triton(BenchmarkConfiguration):
    system = System("A100-SXM-80GB", Architecture.Ampere, 1)
    active_sms = 100
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/imagenet/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/imagenet/ResNet50/int8_linear"
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 64
    gpu_copy_streams = 1
    gpu_inference_streams = 4
    server_target_qps = 30000
    start_from_device = True
    use_cuda_thread_per_device = True
    use_graphs = True
    scenario = Scenario.Server
    benchmark = Benchmark.ResNet50
    gather_kernel_buffer_threshold = 32
    instance_group_count = 4
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx4(BenchmarkConfiguration):
    _system_alias = "DGX Station A100 - Red October"
    _notes = "This should not inherit from A100_SXM_80GB (DGX-A100), and cannot use start_from_device"

    system = System("A100-SXM-80GB", Architecture.Ampere, 4)
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
    gpu_inference_streams = 4
    server_target_qps = 107000
    use_cuda_thread_per_device = True
    use_graphs = True
    scenario = Scenario.Server
    benchmark = Benchmark.ResNet50


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx4_Triton(BenchmarkConfiguration):
    _system_alias = "DGX Station A100 - Red October"
    _notes = "This should not inherit from A100_SXM_80GB (DGX-A100), and cannot use start_from_device"

    system = System("A100-SXM-80GB", Architecture.Ampere, 4)
    active_sms = 100
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/imagenet/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/imagenet/ResNet50/int8_linear"
    use_deque_limit = True
    deque_timeout_usec = 4000
    gpu_batch_size = 64
    gpu_copy_streams = 1
    gpu_inference_streams = 2
    instance_group_count = 2
    batch_triton_requests = False
    server_target_qps = 107000
    use_concurrent_harness = True
    gather_kernel_buffer_threshold = 0
    num_concurrent_batchers = 8
    num_concurrent_issuers = 8
    use_cuda_thread_per_device = True
    request_timeout_usec = 2000
    use_graphs = True
    scenario = Scenario.Server
    benchmark = Benchmark.ResNet50
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_SXM_80GBx4_MaxQ(BenchmarkConfiguration):
    _system_alias = "DGX Station A100 - Red October"
    _notes = "This should not inherit from A100_SXM_80GB (DGX-A100), and cannot use start_from_device"

    system = System("A100-SXM-80GB", Architecture.Ampere, 4)
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
    gpu_inference_streams = 4
    server_target_qps = 107000
    use_cuda_thread_per_device = True
    use_graphs = True
    scenario = Scenario.Server
    benchmark = Benchmark.ResNet50
    power_limit = 225


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_SXM_80GBx4_Triton_MaxQ(BenchmarkConfiguration):
    _system_alias = "DGX Station A100 - Red October"
    _notes = "This should not inherit from A100_SXM_80GB (DGX-A100), and cannot use start_from_device"

    system = System("A100-SXM-80GB", Architecture.Ampere, 4)
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
    gpu_inference_streams = 4
    server_target_qps = 80000
    use_cuda_thread_per_device = True
    use_graphs = False
    scenario = Scenario.Server
    benchmark = Benchmark.ResNet50
    power_limit = 225
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx8(BenchmarkConfiguration):
    system = System("A100-SXM-80GB", Architecture.Ampere, 8)
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
    server_target_qps = 260000
    start_from_device = True
    use_cuda_thread_per_device = True
    use_graphs = True
    scenario = Scenario.Server
    benchmark = Benchmark.ResNet50


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx8_Triton(BenchmarkConfiguration):
    system = System("A100-SXM-80GB", Architecture.Ampere, 8)
    active_sms = 100
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/imagenet/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/imagenet/ResNet50/int8_linear"
    use_deque_limit = True
    deque_timeout_usec = 4000
    gpu_batch_size = 128
    gpu_copy_streams = 1
    gpu_inference_streams = 2
    instance_group_count = 2
    batch_triton_requests = False
    server_target_qps = 220000
    use_concurrent_harness = True
    gather_kernel_buffer_threshold = 0
    num_concurrent_batchers = 8
    num_concurrent_issuers = 8
    use_cuda_thread_per_device = True
    request_timeout_usec = 2000
    use_graphs = True
    scenario = Scenario.Server
    benchmark = Benchmark.ResNet50
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_SXM_80GBx8_MaxQ(BenchmarkConfiguration):
    system = System("A100-SXM-80GB", Architecture.Ampere, 8)
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
    server_target_qps = 232000
    start_from_device = True
    use_cuda_thread_per_device = True
    use_graphs = True
    scenario = Scenario.Server
    benchmark = Benchmark.ResNet50
    power_limit = 225


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_SXM_80GBx8_Triton_MaxQ(BenchmarkConfiguration):
    system = System("A100-SXM-80GB", Architecture.Ampere, 8)
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
    server_target_qps = 200000
    use_cuda_thread_per_device = True
    use_graphs = False
    scenario = Scenario.Server
    benchmark = Benchmark.ResNet50
    power_limit = 225
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_40GBx1(BenchmarkConfiguration):
    system = System("A100-SXM4-40GB", Architecture.Ampere, 1)
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
    gpu_inference_streams = 4
    server_target_qps = 30800
    start_from_device = True
    use_cuda_thread_per_device = True
    use_graphs = True
    scenario = Scenario.Server
    benchmark = Benchmark.ResNet50


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_40GBx1_Triton(BenchmarkConfiguration):
    system = System("A100-SXM4-40GB", Architecture.Ampere, 1)
    active_sms = 100
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/imagenet/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/imagenet/ResNet50/int8_linear"
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 64
    gpu_copy_streams = 1
    gpu_inference_streams = 4
    server_target_qps = 30000
    start_from_device = True
    use_cuda_thread_per_device = True
    use_graphs = True
    scenario = Scenario.Server
    benchmark = Benchmark.ResNet50
    gather_kernel_buffer_threshold = 32
    instance_group_count = 4
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_40GBx8(BenchmarkConfiguration):
    system = System("A100-SXM4-40GB", Architecture.Ampere, 8)
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
    server_target_qps = 255000
    start_from_device = True
    use_cuda_thread_per_device = True
    use_graphs = True
    scenario = Scenario.Server
    benchmark = Benchmark.ResNet50


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_40GBx8_Triton(BenchmarkConfiguration):
    system = System("A100-SXM4-40GB", Architecture.Ampere, 8)
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
    server_target_qps = 160000
    use_cuda_thread_per_device = True
    use_graphs = False
    scenario = Scenario.Server
    benchmark = Benchmark.ResNet50
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A10x1(BenchmarkConfiguration):
    system = System("A10", Architecture.Ampere, 1)
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
    server_target_qps = 11000
    use_cuda_thread_per_device = True
    use_graphs = True
    scenario = Scenario.Server
    benchmark = Benchmark.ResNet50


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A10x1_Triton(BenchmarkConfiguration):
    system = System("A10", Architecture.Ampere, 1)
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
    server_target_qps = 10000
    use_cuda_thread_per_device = True
    use_graphs = True
    scenario = Scenario.Server
    benchmark = Benchmark.ResNet50
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A10x8(BenchmarkConfiguration):
    system = System("A10", Architecture.Ampere, 8)
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
    server_target_qps = 88000.0
    use_cuda_thread_per_device = True
    use_graphs = True
    scenario = Scenario.Server
    benchmark = Benchmark.ResNet50


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A10x8_Triton(BenchmarkConfiguration):
    system = System("A10", Architecture.Ampere, 8)
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
    server_target_qps = 83500
    use_cuda_thread_per_device = True
    use_graphs = False
    scenario = Scenario.Server
    benchmark = Benchmark.ResNet50
    use_triton = True
    gather_kernel_buffer_threshold = 64
    max_queue_delay_usec = 500
    request_timeout_usec = 8000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb(BenchmarkConfiguration):
    _mig_configuration = MIGConfiguration({0: {MIGSlice(1, 6): 1}})
    system = System("A30", Architecture.Ampere, 1, mig_conf=_mig_configuration)
    active_sms = 100
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/imagenet/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/imagenet/ResNet50/int8_linear"
    use_deque_limit = True
    deque_timeout_usec = 200
    gpu_batch_size = 8
    gpu_copy_streams = 4
    gpu_inference_streams = 3
    server_target_qps = 3400
    use_cuda_thread_per_device = True
    use_graphs = True
    workspace_size = 1610612736
    scenario = Scenario.Server
    benchmark = Benchmark.ResNet50


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_Hetero(A30_MIG_1x1g6gb):
    server_target_qps = 3100


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_Triton(BenchmarkConfiguration):
    _mig_configuration = MIGConfiguration({0: {MIGSlice(1, 6): 1}})
    system = System("A30", Architecture.Ampere, 1, mig_conf=_mig_configuration)
    active_sms = 100
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/imagenet/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/imagenet/ResNet50/int8_linear"
    use_deque_limit = True
    deque_timeout_usec = 200
    gpu_batch_size = 8
    gpu_copy_streams = 4
    gpu_inference_streams = 3
    server_target_qps = 1800
    use_cuda_thread_per_device = True
    use_graphs = False
    workspace_size = 1610612736
    scenario = Scenario.Server
    benchmark = Benchmark.ResNet50
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
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/imagenet/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/imagenet/ResNet50/int8_linear"
    use_deque_limit = True
    deque_timeout_usec = 1000
    gpu_batch_size = 4
    gpu_copy_streams = 4
    gpu_inference_streams = 3
    server_target_qps = 220000
    use_cuda_thread_per_device = True
    use_graphs = True
    workspace_size = 1610612736
    scenario = Scenario.Server
    benchmark = Benchmark.ResNet50


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
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/imagenet/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/imagenet/ResNet50/int8_linear"
    use_deque_limit = True
    deque_timeout_usec = 1000
    gpu_batch_size = 8
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    server_target_qps = 65000
    workspace_size = 1610612736
    use_graphs = False
    scenario = Scenario.Server
    benchmark = Benchmark.ResNet50
    use_triton = True
    instance_group_count = 1
    max_queue_delay_usec = 1000
    preferred_batch_size = 8


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x1(BenchmarkConfiguration):
    system = System("A30", Architecture.Ampere, 1)
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
    server_target_qps = 15079.999999999998
    use_cuda_thread_per_device = True
    use_graphs = True
    scenario = Scenario.Server
    benchmark = Benchmark.ResNet50


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x1_Triton(BenchmarkConfiguration):
    system = System("A30", Architecture.Ampere, 1)
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
    server_target_qps = 13919.999999999998
    use_cuda_thread_per_device = True
    use_graphs = False
    scenario = Scenario.Server
    benchmark = Benchmark.ResNet50
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x8(BenchmarkConfiguration):
    system = System("A30", Architecture.Ampere, 8)
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
    server_target_qps = 116000
    use_cuda_thread_per_device = True
    use_graphs = True
    scenario = Scenario.Server
    benchmark = Benchmark.ResNet50


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x8_Triton(BenchmarkConfiguration):
    system = System("A30", Architecture.Ampere, 8)
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
    server_target_qps = 110000
    use_cuda_thread_per_device = True
    use_graphs = False
    scenario = Scenario.Server
    benchmark = Benchmark.ResNet50
    use_triton = True
    gather_kernel_buffer_threshold = 64
    max_queue_delay_usec = 1000
    request_timeout_usec = 8000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x1(BenchmarkConfiguration):
    system = System("T4", Architecture.Turing, 1)
    active_sms = 100
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/imagenet/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/imagenet/ResNet50/int8_linear"
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 26
    gpu_copy_streams = 4
    gpu_inference_streams = 2
    server_target_qps = 4650
    use_cuda_thread_per_device = False
    use_graphs = False
    scenario = Scenario.Server
    benchmark = Benchmark.ResNet50


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x1_Triton(BenchmarkConfiguration):
    system = System("T4", Architecture.Turing, 1)
    active_sms = 100
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/imagenet/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/imagenet/ResNet50/int8_linear"
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 26
    gpu_copy_streams = 4
    gpu_inference_streams = 2
    server_target_qps = 4150
    use_cuda_thread_per_device = False
    use_graphs = False
    scenario = Scenario.Server
    benchmark = Benchmark.ResNet50
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x20(BenchmarkConfiguration):
    system = System("T4", Architecture.Turing, 20)
    active_sms = 100
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/imagenet/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/imagenet/ResNet50/int8_linear"
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 52
    gpu_copy_streams = 8
    gpu_inference_streams = 4
    server_target_qps = 101000
    use_cuda_thread_per_device = True
    use_graphs = True
    scenario = Scenario.Server
    benchmark = Benchmark.ResNet50


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x20_Triton(BenchmarkConfiguration):
    system = System("T4", Architecture.Turing, 20)
    active_sms = 100
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/imagenet/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/imagenet/ResNet50/int8_linear"
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 52
    gpu_copy_streams = 8
    gpu_inference_streams = 4
    server_target_qps = 97000
    use_cuda_thread_per_device = True
    use_graphs = True
    scenario = Scenario.Server
    benchmark = Benchmark.ResNet50
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x8(BenchmarkConfiguration):
    system = System("T4", Architecture.Turing, 8)
    active_sms = 100
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/imagenet/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/imagenet/ResNet50/int8_linear"
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 26
    gpu_copy_streams = 8
    gpu_inference_streams = 2
    server_target_qps = 42000
    use_cuda_thread_per_device = True
    use_graphs = True
    scenario = Scenario.Server
    benchmark = Benchmark.ResNet50


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x8_Triton(BenchmarkConfiguration):
    system = System("T4", Architecture.Turing, 8)
    active_sms = 100
    input_dtype = "int8"
    input_format = "linear"
    map_path = "data_maps/imagenet/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/imagenet/ResNet50/int8_linear"
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 26
    gpu_copy_streams = 8
    gpu_inference_streams = 2
    server_target_qps = 41200
    use_cuda_thread_per_device = True
    use_graphs = False
    scenario = Scenario.Server
    benchmark = Benchmark.ResNet50
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class Triton_CPU_2S_6258Rx1_Triton(BenchmarkConfiguration):
    system = System("Triton_CPU_2S_6258R", Architecture.Intel_CPU_x86_64, 1)
    active_sms = 100
    input_dtype = "fp32"
    map_path = "data_maps/imagenet/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/imagenet/ResNet50/fp32_nomean"
    use_deque_limit = True
    scenario = Scenario.Server
    benchmark = Benchmark.ResNet50
    batch_size = 1
    model_name = "resnet50_int8_openvino"
    num_instances = 16
    ov_parameters = {'CPU_THREADS_NUM': '56', 'CPU_THROUGHPUT_STREAMS': '14', 'ENABLE_BATCH_PADDING': 'NO', 'SKIP_OV_DYNAMIC_BATCHSIZE': 'YES'}
    server_target_qps = 2025
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class Triton_CPU_4S_8380Hx1_Triton(BenchmarkConfiguration):
    system = System("Triton_CPU_4S_8380H", Architecture.Intel_CPU_x86_64, 1)
    active_sms = 100
    input_dtype = "fp32"
    map_path = "data_maps/imagenet/val_map.txt"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/imagenet/ResNet50/fp32_nomean"
    use_deque_limit = True
    server_target_qps = 4899.5
    scenario = Scenario.Server
    benchmark = Benchmark.ResNet50
    batch_size = 1
    model_name = "resnet50_int8_openvino"
    num_instances = 32
    ov_parameters = {'CPU_THREADS_NUM': '112', 'CPU_THROUGHPUT_STREAMS': '28', 'ENABLE_BATCH_PADDING': 'NO', 'SKIP_OV_DYNAMIC_BATCHSIZE': 'YES'}
    use_triton = True
