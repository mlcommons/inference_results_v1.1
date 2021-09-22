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
class DSS8440_A30x8(BenchmarkConfiguration):
    system = System("DSS8440_A30", Architecture.Ampere, 8)
    check_contiguity = True
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    gpu_num_bundles = 2
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    use_small_tile_gemm_plugin = True
    complete_threads = 2
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.8
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 262100
    offline_expected_qps = 1061240
    max_pairs_per_staging_thread = 262100
    num_staging_batches = 4
    num_staging_threads = 4
    use_jemalloc = True
    scenario = Scenario.Offline
    benchmark = Benchmark.DLRM


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class DSS8440_A30x8_HighAccuracy(BenchmarkConfiguration):
    system = System("DSS8440_A30", Architecture.Ampere, 8)
    check_contiguity = True
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    gpu_num_bundles = 2
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    use_small_tile_gemm_plugin = True
    complete_threads = 2
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.8
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 315000
    offline_expected_qps = 1061240
    max_pairs_per_staging_thread = 262100
    num_staging_batches = 8
    num_staging_threads = 8
    use_jemalloc = True
    scenario = Scenario.Offline
    benchmark = Benchmark.DLRM


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750xa_A100_PCIE_80GBx4(BenchmarkConfiguration):
    system = System("R750xa_A100-PCIE-80GB", Architecture.Ampere, 4)
    check_contiguity = True
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    gpu_num_bundles = 2
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    use_small_tile_gemm_plugin = True
    complete_threads = 1
    deque_timeout_usec = 1
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 315000
    offline_expected_qps = 1230000
    max_pairs_per_staging_thread = 262100
    num_staging_batches = 4
    num_staging_threads = 4
    use_jemalloc = True
    numa_config = "0-3:0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46&4-7:1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47"
    scenario = Scenario.Offline
    benchmark = Benchmark.DLRM
    start_from_device = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R750xa_A100_PCIE_80GBx4_HighAccuracy(BenchmarkConfiguration):
    system = System("R750xa_A100-PCIE-80GB", Architecture.Ampere, 4)
    check_contiguity = True
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    gpu_num_bundles = 2
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    use_small_tile_gemm_plugin = True
    complete_threads = 1
    deque_timeout_usec = 1
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 315000
    offline_expected_qps = 1230000
    max_pairs_per_staging_thread = 262100
    num_staging_batches = 4
    num_staging_threads = 4
    use_jemalloc = True
    numa_config = "0-3:0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46&4-7:1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47"
    scenario = Scenario.Offline
    benchmark = Benchmark.DLRM
    start_from_device = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R750xa_A100_PCIE_80GBx4_HighAccuracy_Triton(BenchmarkConfiguration):
    system = System("R750xa_A100-PCIE-80GB", Architecture.Ampere, 4)
    check_contiguity = True
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    gpu_num_bundles = 2
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    use_small_tile_gemm_plugin = True
    complete_threads = 1
    deque_timeout_usec = 1
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 315000
    offline_expected_qps = 1600000
    max_pairs_per_staging_thread = 262100
    num_staging_batches = 4
    num_staging_threads = 4
    use_jemalloc = True
    numa_config = "0-3:0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46&4-7:1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47"
    scenario = Scenario.Offline
    benchmark = Benchmark.DLRM
    batch_triton_requests = True
    buffer_manager_thread_count = 0
    use_triton = True
    gather_kernel_buffer_threshold = 64


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750xa_A100_PCIE_80GBx4_Triton(BenchmarkConfiguration):
    system = System("R750xa_A100-PCIE-80GB", Architecture.Ampere, 4)
    check_contiguity = True
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    gpu_num_bundles = 2
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    use_small_tile_gemm_plugin = True
    complete_threads = 1
    deque_timeout_usec = 1
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 315000
    offline_expected_qps = 1600000
    max_pairs_per_staging_thread = 262100
    num_staging_batches = 4
    num_staging_threads = 4
    use_jemalloc = True
    numa_config = "0-3:0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46&4-7:1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47"
    scenario = Scenario.Offline
    benchmark = Benchmark.DLRM
    batch_triton_requests = True
    buffer_manager_thread_count = 0
    use_triton = True
    gather_kernel_buffer_threshold = 64


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R7525_A100_PCIE_40GBx3(BenchmarkConfiguration):
    system = System("R7525_A100-PCIE-40GB", Architecture.Ampere, 3)
    check_contiguity = True
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    gpu_num_bundles = 2
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    use_small_tile_gemm_plugin = True
    complete_threads = 1
    deque_timeout_usec = 1
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 315000
    offline_expected_qps = 270000*3
    max_pairs_per_staging_thread = 262100
    num_staging_batches = 8
    num_staging_threads = 8
    use_jemalloc = True
    scenario = Scenario.Offline
    benchmark = Benchmark.DLRM

    
@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R7525_A100_PCIE_40GBx3_HighAccuracy(BenchmarkConfiguration):
    system = System("R7525_A100-PCIE-40GB", Architecture.Ampere, 3)
    check_contiguity = True
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    gpu_num_bundles = 2
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    use_small_tile_gemm_plugin = True
    complete_threads = 1
    deque_timeout_usec = 1
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 315000
    offline_expected_qps = 270000*3
    max_pairs_per_staging_thread = 262100
    num_staging_batches = 8
    num_staging_threads = 8
    use_jemalloc = True
    scenario = Scenario.Offline
    benchmark = Benchmark.DLRM


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R7525_A30x3(BenchmarkConfiguration):
    system = System("R7525_A30", Architecture.Ampere, 3)
    check_contiguity = True
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    gpu_num_bundles = 2
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    use_small_tile_gemm_plugin = True
    complete_threads = 2
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.8
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 262100
    offline_expected_qps = 140000*3
    max_pairs_per_staging_thread = 262100
    num_staging_batches = 4
    num_staging_threads = 4
    use_jemalloc = True
    scenario = Scenario.Offline
    benchmark = Benchmark.DLRM


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R7525_A30x3_HighAccuracy(BenchmarkConfiguration):
    system = System("R7525_A30", Architecture.Ampere, 3)
    check_contiguity = True
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    gpu_num_bundles = 2
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    use_small_tile_gemm_plugin = True
    complete_threads = 2
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.8
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 262100
    offline_expected_qps = 140000*3
    max_pairs_per_staging_thread = 262100
    num_staging_batches = 4
    num_staging_threads = 4
    use_jemalloc = True
    scenario = Scenario.Offline
    benchmark = Benchmark.DLRM

    
@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE2420_A10x2(BenchmarkConfiguration):
    system = System("XE2420_A10", Architecture.Ampere, 2)
    check_contiguity = True
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    gpu_num_bundles = 2
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    use_small_tile_gemm_plugin = True
    complete_threads = 2
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.8
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 316805
    offline_expected_qps = 234442
    max_pairs_per_staging_thread = 262100
    num_staging_batches = 4
    num_staging_threads = 4
    use_jemalloc = True
    scenario = Scenario.Offline
    benchmark = Benchmark.DLRM


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class XE2420_A10x2_MaxQ(XE2420_A10x2):
    pass


# inherits XE2420_A10x2-99 values
@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE2420_A10x2_HighAccuracy(XE2420_A10x2):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class XE2420_A10x2_HighAccuracy_MaxQ(XE2420_A10x2_HighAccuracy):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8545_A100_SXM_80GBx4(BenchmarkConfiguration):
    system = System("XE8545_A100-SXM-80GB", Architecture.Ampere, 4)
    check_contiguity = True
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    gpu_num_bundles = 2
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    use_small_tile_gemm_plugin = True
    complete_threads = 1
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 334000
    offline_expected_qps = 1390000
    max_pairs_per_staging_thread = 262100
    start_from_device = True
    use_jemalloc = True
    scenario = Scenario.Offline
    benchmark = Benchmark.DLRM


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE8545_A100_SXM_80GBx4_HighAccuracy(BenchmarkConfiguration):
    system = System("XE8545_A100-SXM-80GB", Architecture.Ampere, 4)
    check_contiguity = True
    coalesced_tensor = True
    enable_interleaved_top_mlp = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    gpu_num_bundles = 2
    input_dtype = "int8"
    input_format = "chw4"
    output_padding_granularity = 128
    precision = "int8"
    sample_partition_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/sample_partition.npy"
    tensor_path = "${PREPROCESSED_DATA_DIR}/criteo/full_recalib/numeric_int8_chw4.npy,${PREPROCESSED_DATA_DIR}/criteo/full_recalib/categorical_int32.npy"
    use_graphs = False
    use_small_tile_gemm_plugin = True
    complete_threads = 1
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 334000
    offline_expected_qps = 1390000
    max_pairs_per_staging_thread = 262100
    start_from_device = True
    use_jemalloc = True
    scenario = Scenario.Offline
    benchmark = Benchmark.DLRM
