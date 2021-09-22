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
class A100_SXM_80GBx4(BenchmarkConfiguration):
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
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 262100
    gpu_num_bundles = 2
    server_target_qps = 1000000
    start_from_device = False
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM
    numa_config = "3:0-15,64-79&2:16-31,80-95&1:32-47,96-111&0:48-63,112-127"


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GBx4_HighAccuracy(BenchmarkConfiguration):
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
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 262100
    gpu_num_bundles = 2
    server_target_qps = 1000000
    start_from_device = False
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM
    numa_config = "3:0-15,64-79&2:16-31,80-95&1:32-47,96-111&0:48-63,112-127"


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GBx4_HighAccuracy_Triton(BenchmarkConfiguration):
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
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 262100
    gpu_num_bundles = 2
    server_target_qps = 362500
    start_from_device = False
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
class A100_SXM_80GBx4_Triton(BenchmarkConfiguration):
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
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 262100
    gpu_num_bundles = 2
    server_target_qps = 362500
    start_from_device = False
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM
    max_queue_delay_usec = 1000
    use_triton = True
    batch_triton_requests = True
    gather_kernel_buffer_threshold = 64
    request_timeout_usec = 2000


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
    start_from_device = False
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
    start_from_device = False
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
    start_from_device = False
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
    start_from_device = False
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.DLRM
    max_queue_delay_usec = 1000
    use_triton = True
    batch_triton_requests = True
    gather_kernel_buffer_threshold = 64
    request_timeout_usec = 2000
