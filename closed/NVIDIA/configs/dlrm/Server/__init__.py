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
class A100_PCIex1(BenchmarkConfiguration):
    system = System("A100-PCIe", Architecture.Ampere, 1)
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    deque_timeout_usec = 1
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 274000
    gpu_num_bundles = 2
    num_staging_batches = 4
    num_staging_threads = 4
    server_target_qps = 180000
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIex1_HighAccuracy(A100_PCIex1):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIex1_Triton(BenchmarkConfiguration):
    system = System("A100-PCIe", Architecture.Ampere, 1)
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    deque_timeout_usec = 1
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 274000
    gpu_num_bundles = 2
    num_staging_batches = 4
    num_staging_threads = 4
    server_target_qps = 180000
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM
    batch_triton_requests = True
    buffer_manager_thread_count = 8
    max_queue_delay_usec = 10000
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIex1_HighAccuracy_Triton(A100_PCIex1_Triton):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIex8(BenchmarkConfiguration):
    system = System("A100-PCIe", Architecture.Ampere, 8)
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    deque_timeout_usec = 1
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 274000
    gpu_num_bundles = 2
    num_staging_batches = 4
    num_staging_threads = 4
    server_target_qps = 800000
    use_jemalloc = False
    use_small_tile_gemm_plugin = True
    numa_config = "3:0-15&2:16-31&1:32-47&0:48-63&7:64-79&6:80-95&5:96-111&4:112-127"
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIex8_HighAccuracy(A100_PCIex8):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIex8_Triton(BenchmarkConfiguration):
    system = System("A100-PCIe", Architecture.Ampere, 8)
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    deque_timeout_usec = 1
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 274000
    gpu_num_bundles = 2
    num_staging_batches = 4
    num_staging_threads = 4
    server_target_qps = 600000
    use_jemalloc = False
    use_small_tile_gemm_plugin = True
    numa_config = "3:0-15&2:16-31&1:32-47&0:48-63&7:64-79&6:80-95&5:96-111&4:112-127"
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM
    batch_triton_requests = True
    buffer_manager_thread_count = 0
    max_queue_delay_usec = 10000
    use_triton = True
    gather_kernel_buffer_threshold = 64


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIex8_HighAccuracy_Triton(A100_PCIex8_Triton):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_PCIex8_MaxQ(BenchmarkConfiguration):
    system = System("A100-PCIe", Architecture.Ampere, 8)
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    deque_timeout_usec = 1
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 274000
    gpu_num_bundles = 2
    num_staging_batches = 4
    num_staging_threads = 4
    server_target_qps = 750000
    use_jemalloc = False
    use_small_tile_gemm_plugin = True
    numa_config = "3:0-7&2:8-15&1:16-23&0:24-31&7:32-39&6:40-47&5:48-55&4:56-63"
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM
    power_limit = 225


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class A100_PCIex8_HighAccuracy_MaxQ(A100_PCIex8_MaxQ):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class A100_PCIex8_HighAccuracy_Triton_MaxQ(BenchmarkConfiguration):
    system = System("A100-PCIe", Architecture.Ampere, 8)
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    deque_timeout_usec = 1
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 274000
    gpu_num_bundles = 2
    num_staging_batches = 4
    num_staging_threads = 4
    server_target_qps = 700000
    use_jemalloc = False
    use_small_tile_gemm_plugin = True
    numa_config = "3:0-7&2:8-15&1:16-23&0:24-31&7:32-39&6:40-47&5:48-55&4:56-63"
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM
    buffer_manager_thread_count = 8
    max_queue_delay_usec = 10000
    power_limit = 225
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_PCIex8_Triton_MaxQ(BenchmarkConfiguration):
    system = System("A100-PCIe", Architecture.Ampere, 8)
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    deque_timeout_usec = 1
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 274000
    gpu_num_bundles = 2
    num_staging_batches = 4
    num_staging_threads = 4
    server_target_qps = 700000
    use_jemalloc = False
    use_small_tile_gemm_plugin = True
    numa_config = "3:0-7&2:8-15&1:16-23&0:24-31&7:32-39&6:40-47&5:48-55&4:56-63"
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM
    buffer_manager_thread_count = 8
    max_queue_delay_usec = 10000
    power_limit = 225
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx1(BenchmarkConfiguration):
    system = System("A100-PCIe-80GB", Architecture.Ampere, 1)
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    deque_timeout_usec = 1
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 274000
    gpu_num_bundles = 2
    num_staging_batches = 4
    num_staging_threads = 4
    server_target_qps = 180000
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_80GBx1_HighAccuracy(A100_PCIe_80GBx1):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx1_Triton(BenchmarkConfiguration):
    system = System("A100-PCIe-80GB", Architecture.Ampere, 1)
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    deque_timeout_usec = 1
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 274000
    gpu_num_bundles = 2
    num_staging_batches = 4
    num_staging_threads = 4
    server_target_qps = 180000
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM
    batch_triton_requests = True
    buffer_manager_thread_count = 8
    max_queue_delay_usec = 10000
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_80GBx1_HighAccuracy_Triton(A100_PCIe_80GBx1_Triton):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx8(BenchmarkConfiguration):
    system = System("A100-PCIe-80GB", Architecture.Ampere, 8)
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    deque_timeout_usec = 1
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 274000
    gpu_num_bundles = 2
    num_staging_batches = 4
    num_staging_threads = 4
    server_target_qps = 1300000
    use_jemalloc = False
    use_small_tile_gemm_plugin = True
    numa_config = "3:0-15&2:16-31&1:32-47&0:48-63&7:64-79&6:80-95&5:96-111&4:112-127"
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_80GBx8_HighAccuracy(A100_PCIe_80GBx8):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx8_Triton(BenchmarkConfiguration):
    system = System("A100-PCIe-80GB", Architecture.Ampere, 8)
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    deque_timeout_usec = 1
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 274000
    gpu_num_bundles = 2
    num_staging_batches = 4
    num_staging_threads = 4
    server_target_qps = 600000
    use_jemalloc = False
    use_small_tile_gemm_plugin = True
    numa_config = "3:0-15&2:16-31&1:32-47&0:48-63&7:64-79&6:80-95&5:96-111&4:112-127"
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM
    batch_triton_requests = True
    buffer_manager_thread_count = 0
    max_queue_delay_usec = 10000
    use_triton = True
    gather_kernel_buffer_threshold = 64


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_80GBx8_HighAccuracy_Triton(A100_PCIe_80GBx8_Triton):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x1(BenchmarkConfiguration):
    system = System("A100-PCIe-80GB", Architecture.Ampere, 1, cpu_arch=CPUArch.aarch64)
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    deque_timeout_usec = 1
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 274000
    gpu_num_bundles = 2
    num_staging_batches = 4
    num_staging_threads = 4
    server_target_qps = 180000
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x1_HighAccuracy(A100_PCIe_80GB_aarch64x1):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x2(BenchmarkConfiguration):
    system = System("A100-PCIe-80GB", Architecture.Ampere, 2, cpu_arch=CPUArch.aarch64)
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    deque_timeout_usec = 1
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 274000
    gpu_num_bundles = 2
    num_staging_batches = 4
    num_staging_threads = 4
    server_target_qps = 300000
    use_jemalloc = False
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x2_HighAccuracy(A100_PCIe_80GB_aarch64x2):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x4(BenchmarkConfiguration):
    system = System("A100-PCIe-80GB", Architecture.Ampere, 4, cpu_arch=CPUArch.aarch64)
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    deque_timeout_usec = 1
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 274000
    gpu_num_bundles = 2
    num_staging_batches = 4
    num_staging_threads = 4
    server_target_qps = 600000
    use_jemalloc = False
    use_small_tile_gemm_plugin = True
    # TODO: Set numa
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x4_HighAccuracy(A100_PCIe_80GB_aarch64x4):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_PCIe_80GB_aarch64x4_MaxQ(BenchmarkConfiguration):
    system = System("A100-PCIe-80GB", Architecture.Ampere, 4, cpu_arch=CPUArch.aarch64)
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    deque_timeout_usec = 1
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 274000
    gpu_num_bundles = 2
    num_staging_batches = 4
    num_staging_threads = 4
    server_target_qps = 500000
    use_jemalloc = False
    use_small_tile_gemm_plugin = True
    # TODO: Set numa
    power_limit = 225
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class A100_PCIe_80GB_aarch64x4_HighAccuracy_MaxQ(A100_PCIe_80GB_aarch64x4_MaxQ):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_aarch64x1(BenchmarkConfiguration):
    system = System("A100-PCIe", Architecture.Ampere, 1, cpu_arch=CPUArch.aarch64)
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    deque_timeout_usec = 1
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 274000
    gpu_num_bundles = 2
    num_staging_batches = 4
    num_staging_threads = 4
    server_target_qps = 180000
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_aarch64x1_HighAccuracy(A100_PCIe_aarch64x1):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_aarch64x2(BenchmarkConfiguration):
    system = System("A100-PCIe", Architecture.Ampere, 2, cpu_arch=CPUArch.aarch64)
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    deque_timeout_usec = 1
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 274000
    gpu_num_bundles = 2
    num_staging_batches = 4
    num_staging_threads = 4
    server_target_qps = 300000
    use_jemalloc = False
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_aarch64x2_HighAccuracy(A100_PCIe_aarch64x2):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_aarch64x4(BenchmarkConfiguration):
    system = System("A100-PCIe", Architecture.Ampere, 4, cpu_arch=CPUArch.aarch64)
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    deque_timeout_usec = 1
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 274000
    gpu_num_bundles = 2
    num_staging_batches = 4
    num_staging_threads = 4
    server_target_qps = 670000
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_aarch64x4_HighAccuracy(A100_PCIe_aarch64x4):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_PCIe_aarch64x4_MaxQ(BenchmarkConfiguration):
    system = System("A100-PCIe", Architecture.Ampere, 4, cpu_arch=CPUArch.aarch64)
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    deque_timeout_usec = 1
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 224000
    gpu_num_bundles = 2
    num_staging_batches = 4
    num_staging_threads = 4
    server_target_qps = 650000
    use_jemalloc = False
    use_small_tile_gemm_plugin = True
    power_limit = 225
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class A100_PCIe_aarch64x4_HighAccuracy_MaxQ(A100_PCIe_aarch64x4_MaxQ):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb(BenchmarkConfiguration):
    _mig_configuration = MIGConfiguration({0: {MIGSlice(1, 10): 1}})
    system = System("A100-SXM-80GB", Architecture.Ampere, 1, mig_conf=_mig_configuration)
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.3
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 65500
    gpu_num_bundles = 2
    num_staging_batches = 4
    num_staging_threads = 4
    server_target_qps = 36000
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb_Hetero(A100_SXM_80GB_MIG_1x1g10gb):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb_HighAccuracy(BenchmarkConfiguration):
    _mig_configuration = MIGConfiguration({0: {MIGSlice(1, 10): 1}})
    system = System("A100-SXM-80GB", Architecture.Ampere, 1, mig_conf=_mig_configuration)
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.3
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 65500
    gpu_num_bundles = 2
    num_staging_batches = 4
    num_staging_threads = 4
    server_target_qps = 36000
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb_Hetero_HighAccuracy(A100_SXM_80GB_MIG_1x1g10gb_HighAccuracy):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb_HighAccuracy_Triton(BenchmarkConfiguration):
    _mig_configuration = MIGConfiguration({0: {MIGSlice(1, 10): 1}})
    system = System("A100-SXM-80GB", Architecture.Ampere, 1, mig_conf=_mig_configuration)
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.3
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 65500
    gpu_num_bundles = 2
    num_staging_batches = 4
    num_staging_threads = 4
    server_target_qps = 20000
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM
    buffer_manager_thread_count = 8
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb_Triton(BenchmarkConfiguration):
    _mig_configuration = MIGConfiguration({0: {MIGSlice(1, 10): 1}})
    system = System("A100-SXM-80GB", Architecture.Ampere, 1, mig_conf=_mig_configuration)
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.3
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 65500
    gpu_num_bundles = 2
    num_staging_batches = 4
    num_staging_threads = 4
    server_target_qps = 20000
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM
    buffer_manager_thread_count = 8
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx1(BenchmarkConfiguration):
    system = System("A100-SXM-80GB", Architecture.Ampere, 1)
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    deque_timeout_usec = 1
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 262100
    gpu_num_bundles = 2
    num_staging_batches = 4
    num_staging_threads = 4
    server_target_qps = 286000
    start_from_device = True
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GBx1_HighAccuracy(BenchmarkConfiguration):
    system = System("A100-SXM-80GB", Architecture.Ampere, 1)
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    deque_timeout_usec = 1
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 262100
    gpu_num_bundles = 2
    num_staging_batches = 4
    num_staging_threads = 4
    server_target_qps = 286000
    start_from_device = True
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GBx1_HighAccuracy_Triton(BenchmarkConfiguration):
    system = System("A100-SXM-80GB", Architecture.Ampere, 1)
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    deque_timeout_usec = 1
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 262100
    gpu_num_bundles = 2
    num_staging_batches = 4
    num_staging_threads = 4
    server_target_qps = 270000
    start_from_device = True
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM
    batch_triton_requests = True
    buffer_manager_thread_count = 0
    gather_kernel_buffer_threshold = 10
    max_queue_delay_usec = 1
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx1_Triton(BenchmarkConfiguration):
    system = System("A100-SXM-80GB", Architecture.Ampere, 1)
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    deque_timeout_usec = 1
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 262100
    gpu_num_bundles = 2
    num_staging_batches = 4
    num_staging_threads = 4
    server_target_qps = 270000
    start_from_device = True
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM
    batch_triton_requests = True
    buffer_manager_thread_count = 0
    gather_kernel_buffer_threshold = 10
    max_queue_delay_usec = 1
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx4(BenchmarkConfiguration):
    _system_alias = "DGX Station A100 - Red October"
    _notes = "This should not inherit from A100_SXM_80GB (DGX-A100), and cannot use start_from_device"

    system = System("A100-SXM-80GB", Architecture.Ampere, 4)
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    deque_timeout_usec = 1
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 274000
    gpu_num_bundles = 2
    num_staging_batches = 4
    num_staging_threads = 4
    server_target_qps = 950000
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    numa_config = "3:0-15,64-79&2:16-31,80-95&1:32-47,96-111&0:48-63,112-127"
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GBx4_HighAccuracy(BenchmarkConfiguration):
    _system_alias = "DGX Station A100 - Red October"
    _notes = "This should not inherit from A100_SXM_80GB (DGX-A100), and cannot use start_from_device"

    system = System("A100-SXM-80GB", Architecture.Ampere, 4)
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    deque_timeout_usec = 1
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 274000
    gpu_num_bundles = 2
    num_staging_batches = 4
    num_staging_threads = 4
    server_target_qps = 950000
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    numa_config = "3:0-15,64-79&2:16-31,80-95&1:32-47,96-111&0:48-63,112-127"
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GBx4_HighAccuracy_Triton(BenchmarkConfiguration):
    _system_alias = "DGX Station A100 - Red October"
    _notes = "This should not inherit from A100_SXM_80GB (DGX-A100), and cannot use start_from_device"

    system = System("A100-SXM-80GB", Architecture.Ampere, 4)
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    deque_timeout_usec = 1
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 274000
    gpu_num_bundles = 2
    num_staging_batches = 4
    num_staging_threads = 4
    server_target_qps = 750000
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    numa_config = "3:0-15,64-79&2:16-31,80-95&1:32-47,96-111&0:48-63,112-127"
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM
    batch_triton_requests = True
    buffer_manager_thread_count = 0
    max_queue_delay_usec = 1
    use_triton = True
    gather_kernel_buffer_threshold = 10


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx4_Triton(A100_SXM_80GBx4_HighAccuracy_Triton):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_SXM_80GBx4_MaxQ(BenchmarkConfiguration):
    _system_alias = "DGX Station A100 - Red October"
    _notes = "This should not inherit from A100_SXM_80GB (DGX-A100), and cannot use start_from_device"

    system = System("A100-SXM-80GB", Architecture.Ampere, 4)
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    deque_timeout_usec = 1
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 224000
    gpu_num_bundles = 2
    num_staging_batches = 4
    num_staging_threads = 4
    server_target_qps = 890000
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    numa_config = "3:0-7,32-39&2:8-15,40-47&1:16-23,48-55&0:24-31,56-63"
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM
    power_limit = 250


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class A100_SXM_80GBx4_HighAccuracy_MaxQ(BenchmarkConfiguration):
    _system_alias = "DGX Station A100 - Red October"
    _notes = "This should not inherit from A100_SXM_80GB (DGX-A100), and cannot use start_from_device"

    system = System("A100-SXM-80GB", Architecture.Ampere, 4)
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    deque_timeout_usec = 1
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 224000
    gpu_num_bundles = 2
    num_staging_batches = 4
    num_staging_threads = 4
    server_target_qps = 890000
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    numa_config = "3:0-7,32-39&2:8-15,40-47&1:16-23,48-55&0:24-31,56-63"
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM
    power_limit = 250


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class A100_SXM_80GBx4_HighAccuracy_Triton_MaxQ(BenchmarkConfiguration):
    _system_alias = "DGX Station A100 - Red October"
    _notes = "This should not inherit from A100_SXM_80GB (DGX-A100), and cannot use start_from_device"

    system = System("A100-SXM-80GB", Architecture.Ampere, 4)
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    deque_timeout_usec = 1
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 274000
    gpu_num_bundles = 2
    num_staging_batches = 4
    num_staging_threads = 4
    server_target_qps = 270000
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    numa_config = "3:0-7,32-39&2:8-15,40-47&1:16-23,48-55&0:24-31,56-63"
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM
    batch_triton_requests = True
    buffer_manager_thread_count = 8
    max_queue_delay_usec = 10000
    power_limit = 250
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_SXM_80GBx4_Triton_MaxQ(BenchmarkConfiguration):
    _system_alias = "DGX Station A100 - Red October"
    _notes = "This should not inherit from A100_SXM_80GB (DGX-A100), and cannot use start_from_device"

    system = System("A100-SXM-80GB", Architecture.Ampere, 4)
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    deque_timeout_usec = 1
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 274000
    gpu_num_bundles = 2
    num_staging_batches = 4
    num_staging_threads = 4
    server_target_qps = 270000
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    numa_config = "3:0-7,32-39&2:8-15,40-47&1:16-23,48-55&0:24-31,56-63"
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM
    batch_triton_requests = True
    buffer_manager_thread_count = 8
    max_queue_delay_usec = 10000
    power_limit = 250
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx8(BenchmarkConfiguration):
    system = System("A100-SXM-80GB", Architecture.Ampere, 8)
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 262100
    gpu_num_bundles = 2
    server_target_qps = 2300000
    start_from_device = True
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GBx8_HighAccuracy(BenchmarkConfiguration):
    system = System("A100-SXM-80GB", Architecture.Ampere, 8)
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 262100
    gpu_num_bundles = 2
    server_target_qps = 2300000
    start_from_device = True
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GBx8_HighAccuracy_Triton(BenchmarkConfiguration):
    system = System("A100-SXM-80GB", Architecture.Ampere, 8)
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 262100
    gpu_num_bundles = 2
    server_target_qps = 725000
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM
    max_queue_delay_usec = 1000
    use_triton = True
    batch_triton_requests = True
    gather_kernel_buffer_threshold = 64
    request_timeout_usec = 2000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx8_Triton(BenchmarkConfiguration):
    system = System("A100-SXM-80GB", Architecture.Ampere, 8)
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 262100
    gpu_num_bundles = 2
    server_target_qps = 725000
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM
    max_queue_delay_usec = 1000
    use_triton = True
    batch_triton_requests = True
    gather_kernel_buffer_threshold = 64
    request_timeout_usec = 2000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_SXM_80GBx8_MaxQ(BenchmarkConfiguration):
    system = System("A100-SXM-80GB", Architecture.Ampere, 8)
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 262100
    gpu_num_bundles = 2
    server_target_qps = 2000000
    start_from_device = True
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM
    power_limit = 275


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class A100_SXM_80GBx8_HighAccuracy_MaxQ(BenchmarkConfiguration):
    system = System("A100-SXM-80GB", Architecture.Ampere, 8)
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 262100
    gpu_num_bundles = 2
    server_target_qps = 2000000
    start_from_device = True
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM
    power_limit = 275


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_40GBx1(BenchmarkConfiguration):
    system = System("A100-SXM4-40GB", Architecture.Ampere, 1)
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    deque_timeout_usec = 1
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 262100
    gpu_num_bundles = 2
    num_staging_batches = 4
    num_staging_threads = 4
    server_target_qps = 255000
    start_from_device = True
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM4_40GBx1_HighAccuracy(BenchmarkConfiguration):
    system = System("A100-SXM4-40GB", Architecture.Ampere, 1)
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    deque_timeout_usec = 1
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 262100
    gpu_num_bundles = 2
    num_staging_batches = 4
    num_staging_threads = 4
    server_target_qps = 255000
    start_from_device = True
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM4_40GBx1_HighAccuracy_Triton(BenchmarkConfiguration):
    system = System("A100-SXM4-40GB", Architecture.Ampere, 1)
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    deque_timeout_usec = 1
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 262100
    gpu_num_bundles = 2
    num_staging_batches = 4
    num_staging_threads = 4
    server_target_qps = 245000
    start_from_device = True
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM
    batch_triton_requests = True
    buffer_manager_thread_count = 0
    gather_kernel_buffer_threshold = 10
    max_queue_delay_usec = 1
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_40GBx1_Triton(BenchmarkConfiguration):
    system = System("A100-SXM4-40GB", Architecture.Ampere, 1)
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    deque_timeout_usec = 1
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 262100
    gpu_num_bundles = 2
    num_staging_batches = 4
    num_staging_threads = 4
    server_target_qps = 245000
    start_from_device = True
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM
    batch_triton_requests = True
    buffer_manager_thread_count = 0
    gather_kernel_buffer_threshold = 10
    max_queue_delay_usec = 1
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_40GBx8(BenchmarkConfiguration):
    system = System("A100-SXM4-40GB", Architecture.Ampere, 8)
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 262100
    gpu_num_bundles = 2
    server_target_qps = 2100000
    start_from_device = True
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM4_40GBx8_HighAccuracy(BenchmarkConfiguration):
    system = System("A100-SXM4-40GB", Architecture.Ampere, 8)
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 262100
    gpu_num_bundles = 2
    server_target_qps = 2100000
    start_from_device = True
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM4_40GBx8_HighAccuracy_Triton(BenchmarkConfiguration):
    system = System("A100-SXM4-40GB", Architecture.Ampere, 8)
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 262100
    gpu_num_bundles = 2
    server_target_qps = 80000
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM
    max_queue_delay_usec = 10000
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_40GBx8_Triton(BenchmarkConfiguration):
    system = System("A100-SXM4-40GB", Architecture.Ampere, 8)
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 262100
    gpu_num_bundles = 2
    server_target_qps = 80000
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM
    max_queue_delay_usec = 10000
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A10x1(BenchmarkConfiguration):
    system = System("A10", Architecture.Ampere, 1)
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.8
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 65500
    gpu_num_bundles = 2
    num_staging_batches = 2
    num_staging_threads = 4
    server_target_qps = 68000
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A10x1_HighAccuracy(BenchmarkConfiguration):
    system = System("A10", Architecture.Ampere, 1)
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.8
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 65500
    gpu_num_bundles = 2
    num_staging_batches = 2
    num_staging_threads = 4
    server_target_qps = 68000
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A10x1_HighAccuracy_Triton(BenchmarkConfiguration):
    system = System("A10", Architecture.Ampere, 1)
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.8
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 65500
    gpu_num_bundles = 2
    num_staging_batches = 2
    num_staging_threads = 4
    server_target_qps = 66000
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM
    buffer_manager_thread_count = 0
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A10x1_Triton(BenchmarkConfiguration):
    system = System("A10", Architecture.Ampere, 1)
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.8
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 65500
    gpu_num_bundles = 2
    num_staging_batches = 2
    num_staging_threads = 4
    server_target_qps = 66000
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM
    buffer_manager_thread_count = 0
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A10x8(BenchmarkConfiguration):
    system = System("A10", Architecture.Ampere, 8)
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.8
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 60000
    gpu_num_bundles = 2
    num_staging_batches = 8
    num_staging_threads = 8
    server_target_qps = 680000
    use_jemalloc = False
    use_small_tile_gemm_plugin = True
    numa_config = "0-3:0-27,56-83&4-7:28-55,84-111"
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A10x8_HighAccuracy(BenchmarkConfiguration):
    system = System("A10", Architecture.Ampere, 8)
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.8
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 60000
    gpu_num_bundles = 2
    num_staging_batches = 8
    num_staging_threads = 8
    server_target_qps = 680000
    use_jemalloc = False
    use_small_tile_gemm_plugin = True
    numa_config = "0-3:0-27,56-83&4-7:28-55,84-111"
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A10x8_HighAccuracy_Triton(BenchmarkConfiguration):
    system = System("A10", Architecture.Ampere, 8)
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.8
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 60000
    gpu_num_bundles = 2
    num_staging_batches = 8
    num_staging_threads = 8
    server_target_qps = 500000
    use_jemalloc = False
    use_small_tile_gemm_plugin = True
    numa_config = "0-3:0-27,56-83&4-7:28-55,84-111"
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM
    buffer_manager_thread_count = 0
    use_triton = True
    batch_triton_requests = True
    gather_kernel_buffer_threshold = 64


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A10x8_Triton(BenchmarkConfiguration):
    system = System("A10", Architecture.Ampere, 8)
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.8
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 60000
    gpu_num_bundles = 2
    num_staging_batches = 8
    num_staging_threads = 8
    server_target_qps = 500000
    use_jemalloc = False
    use_small_tile_gemm_plugin = True
    numa_config = "0-3:0-27,56-83&4-7:28-55,84-111"
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM
    buffer_manager_thread_count = 0
    use_triton = True
    batch_triton_requests = True
    gather_kernel_buffer_threshold = 64


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb(BenchmarkConfiguration):
    _mig_configuration = MIGConfiguration({0: {MIGSlice(1, 6): 1}})
    system = System("A30", Architecture.Ampere, 1, mig_conf=_mig_configuration)
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.02
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 226000
    gpu_num_bundles = 2
    num_staging_batches = 2
    num_staging_threads = 4
    server_target_qps = 31000
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_HighAccuracy(BenchmarkConfiguration):
    _mig_configuration = MIGConfiguration({0: {MIGSlice(1, 6): 1}})
    system = System("A30", Architecture.Ampere, 1, mig_conf=_mig_configuration)
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.02
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 226000
    gpu_num_bundles = 2
    num_staging_batches = 2
    num_staging_threads = 4
    server_target_qps = 31000
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_Hetero(A30_MIG_1x1g6gb):
    server_target_qps = 30000


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_Hetero_HighAccuracy(A30_MIG_1x1g6gb_Hetero):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_HighAccuracy_Triton(BenchmarkConfiguration):
    _mig_configuration = MIGConfiguration({0: {MIGSlice(1, 6): 1}})
    system = System("A30", Architecture.Ampere, 1, mig_conf=_mig_configuration)
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.02
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 226000
    gpu_num_bundles = 2
    num_staging_batches = 2
    num_staging_threads = 4
    server_target_qps = 25000.0
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM
    batch_triton_requests = True
    gather_kernel_buffer_threshold = 64
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_Triton(BenchmarkConfiguration):
    _mig_configuration = MIGConfiguration({0: {MIGSlice(1, 6): 1}})
    system = System("A30", Architecture.Ampere, 1, mig_conf=_mig_configuration)
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.02
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 226000
    gpu_num_bundles = 2
    num_staging_batches = 2
    num_staging_threads = 4
    server_target_qps = 25000.0
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM
    batch_triton_requests = True
    gather_kernel_buffer_threshold = 64
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x1(BenchmarkConfiguration):
    system = System("A30", Architecture.Ampere, 1)
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.8
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 226000
    gpu_num_bundles = 2
    num_staging_batches = 2
    num_staging_threads = 4
    server_target_qps = 132000
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30x1_HighAccuracy(BenchmarkConfiguration):
    system = System("A30", Architecture.Ampere, 1)
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.8
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 226000
    gpu_num_bundles = 2
    num_staging_batches = 2
    num_staging_threads = 4
    server_target_qps = 132000
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30x1_HighAccuracy_Triton(BenchmarkConfiguration):
    system = System("A30", Architecture.Ampere, 1)
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.8
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 226000
    gpu_num_bundles = 2
    num_staging_batches = 2
    num_staging_threads = 4
    server_target_qps = 100000
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM
    batch_triton_requests = True
    gather_kernel_buffer_threshold = 64
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x1_Triton(BenchmarkConfiguration):
    system = System("A30", Architecture.Ampere, 1)
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.8
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 226000
    gpu_num_bundles = 2
    num_staging_batches = 2
    num_staging_threads = 4
    server_target_qps = 100000
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM
    batch_triton_requests = True
    gather_kernel_buffer_threshold = 64
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x8(BenchmarkConfiguration):
    system = System("A30", Architecture.Ampere, 8)
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.8
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 131000
    gpu_num_bundles = 2
    num_staging_batches = 8
    num_staging_threads = 8
    server_target_qps = 1000000
    use_jemalloc = False
    use_small_tile_gemm_plugin = True
    numa_config = "3:0-15&2:16-31&1:32-47&0:48-63&7:64-79&6:80-95&5:96-111&4:112-127"
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30x8_HighAccuracy(BenchmarkConfiguration):
    system = System("A30", Architecture.Ampere, 8)
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.8
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 131000
    gpu_num_bundles = 2
    num_staging_batches = 8
    num_staging_threads = 8
    server_target_qps = 1000000
    use_jemalloc = False
    use_small_tile_gemm_plugin = True
    numa_config = "3:0-15&2:16-31&1:32-47&0:48-63&7:64-79&6:80-95&5:96-111&4:112-127"
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30x8_HighAccuracy_Triton(BenchmarkConfiguration):
    system = System("A30", Architecture.Ampere, 8)
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.8
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 131000
    gpu_num_bundles = 2
    num_staging_batches = 8
    num_staging_threads = 8
    server_target_qps = 600000
    use_jemalloc = False
    use_small_tile_gemm_plugin = True
    numa_config = "3:0-15&2:16-31&1:32-47&0:48-63&7:64-79&6:80-95&5:96-111&4:112-127"
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM
    batch_triton_requests = True
    gather_kernel_buffer_threshold = 64
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x8_Triton(BenchmarkConfiguration):
    system = System("A30", Architecture.Ampere, 8)
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.8
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 131000
    gpu_num_bundles = 2
    num_staging_batches = 8
    num_staging_threads = 8
    server_target_qps = 600000
    use_jemalloc = False
    use_small_tile_gemm_plugin = True
    numa_config = "3:0-15&2:16-31&1:32-47&0:48-63&7:64-79&6:80-95&5:96-111&4:112-127"
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM
    batch_triton_requests = True
    gather_kernel_buffer_threshold = 64
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x1(BenchmarkConfiguration):
    system = System("T4", Architecture.Turing, 1)
    coalesced_tensor = True
    enable_interleaved_top_mlp = True
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.5
    gpu_batch_size = 65500
    gpu_num_bundles = 2
    num_staging_batches = 2
    num_staging_threads = 4
    server_target_qps = 24000
    use_jemalloc = True
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class T4x1_HighAccuracy(BenchmarkConfiguration):
    system = System("T4", Architecture.Turing, 1)
    coalesced_tensor = True
    enable_interleaved_top_mlp = True
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.5
    gpu_batch_size = 65500
    gpu_num_bundles = 2
    num_staging_batches = 2
    num_staging_threads = 4
    server_target_qps = 24000
    use_jemalloc = True
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class T4x1_HighAccuracy_Triton(BenchmarkConfiguration):
    system = System("T4", Architecture.Turing, 1)
    coalesced_tensor = True
    enable_interleaved_top_mlp = True
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.5
    gpu_batch_size = 65500
    gpu_num_bundles = 2
    num_staging_batches = 2
    num_staging_threads = 4
    server_target_qps = 24000
    use_jemalloc = True
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM
    buffer_manager_thread_count = 8
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x1_Triton(BenchmarkConfiguration):
    system = System("T4", Architecture.Turing, 1)
    coalesced_tensor = True
    enable_interleaved_top_mlp = True
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.5
    gpu_batch_size = 65500
    gpu_num_bundles = 2
    num_staging_batches = 2
    num_staging_threads = 4
    server_target_qps = 24000
    use_jemalloc = True
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM
    buffer_manager_thread_count = 8
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x20(BenchmarkConfiguration):
    system = System("T4", Architecture.Turing, 20)
    coalesced_tensor = True
    enable_interleaved_top_mlp = True
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    embedding_weights_on_gpu_part = 0.5
    gpu_batch_size = 65800
    gpu_num_bundles = 2
    num_staging_batches = 16
    num_staging_threads = 8
    server_target_qps = 600000
    use_jemalloc = False
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class T4x20_HighAccuracy(BenchmarkConfiguration):
    system = System("T4", Architecture.Turing, 20)
    coalesced_tensor = True
    enable_interleaved_top_mlp = True
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    embedding_weights_on_gpu_part = 0.5
    gpu_batch_size = 65800
    gpu_num_bundles = 2
    num_staging_batches = 16
    num_staging_threads = 8
    server_target_qps = 600000
    use_jemalloc = False
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class T4x20_HighAccuracy_Triton(BenchmarkConfiguration):
    system = System("T4", Architecture.Turing, 20)
    coalesced_tensor = True
    enable_interleaved_top_mlp = True
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    embedding_weights_on_gpu_part = 0.5
    gpu_batch_size = 65800
    gpu_num_bundles = 2
    num_staging_batches = 16
    num_staging_threads = 8
    server_target_qps = 60000
    use_jemalloc = False
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x20_Triton(BenchmarkConfiguration):
    system = System("T4", Architecture.Turing, 20)
    coalesced_tensor = True
    enable_interleaved_top_mlp = True
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    embedding_weights_on_gpu_part = 0.5
    gpu_batch_size = 65800
    gpu_num_bundles = 2
    num_staging_batches = 16
    num_staging_threads = 8
    server_target_qps = 60000
    use_jemalloc = False
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x8(BenchmarkConfiguration):
    system = System("T4", Architecture.Turing, 8)
    coalesced_tensor = True
    enable_interleaved_top_mlp = True
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.5
    gpu_batch_size = 65500
    gpu_num_bundles = 1
    num_staging_batches = 8
    num_staging_threads = 4
    server_target_qps = 250000
    use_jemalloc = True
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class T4x8_HighAccuracy(BenchmarkConfiguration):
    system = System("T4", Architecture.Turing, 8)
    coalesced_tensor = True
    enable_interleaved_top_mlp = True
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.5
    gpu_batch_size = 65500
    gpu_num_bundles = 1
    num_staging_batches = 8
    num_staging_threads = 4
    server_target_qps = 250000
    use_jemalloc = True
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class T4x8_HighAccuracy_Triton(BenchmarkConfiguration):
    system = System("T4", Architecture.Turing, 8)
    coalesced_tensor = True
    enable_interleaved_top_mlp = True
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.5
    gpu_batch_size = 65500
    gpu_num_bundles = 1
    num_staging_batches = 8
    num_staging_threads = 4
    server_target_qps = 55000
    use_jemalloc = True
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x8_Triton(BenchmarkConfiguration):
    system = System("T4", Architecture.Turing, 8)
    coalesced_tensor = True
    enable_interleaved_top_mlp = True
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    complete_threads = 1
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.5
    gpu_batch_size = 65500
    gpu_num_bundles = 1
    num_staging_batches = 8
    num_staging_threads = 4
    server_target_qps = 55000
    use_jemalloc = True
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM
    use_triton = True
