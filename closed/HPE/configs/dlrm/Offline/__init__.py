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
    offline_expected_qps = 2400000
    max_pairs_per_staging_thread = 262100
    start_from_device = False
    use_jemalloc = True
    scenario = Scenario.Offline
    benchmark = Benchmark.DLRM


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GBx4_HighAccuracy(BenchmarkConfiguration):
    system = System("A100-SXM-80GB", Architecture.Ampere, 4)
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
    offline_expected_qps = 2400000
    max_pairs_per_staging_thread = 262100
    start_from_device = False
    use_jemalloc = True
    scenario = Scenario.Offline
    benchmark = Benchmark.DLRM


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GBx4_HighAccuracy_Triton(BenchmarkConfiguration):
    system = System("A100-SXM-80GB", Architecture.Ampere, 4)
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
    offline_expected_qps = 2200000
    max_pairs_per_staging_thread = 262100
    start_from_device = False
    use_jemalloc = True
    scenario = Scenario.Offline
    benchmark = Benchmark.DLRM
    batch_triton_requests = True
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx4_Triton(BenchmarkConfiguration):
    system = System("A100-SXM-80GB", Architecture.Ampere, 4)
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
    gpu_batch_size = 262100
    offline_expected_qps = 2450000
    max_pairs_per_staging_thread = 262100
    start_from_device = False
    use_jemalloc = True
    scenario = Scenario.Offline
    benchmark = Benchmark.DLRM
    batch_triton_requests = True
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx8(BenchmarkConfiguration):
    system = System("A100-SXM-80GB", Architecture.Ampere, 8)
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
    offline_expected_qps = 2400000
    max_pairs_per_staging_thread = 262100
    start_from_device = False
    use_jemalloc = True
    scenario = Scenario.Offline
    benchmark = Benchmark.DLRM


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GBx8_HighAccuracy(BenchmarkConfiguration):
    system = System("A100-SXM-80GB", Architecture.Ampere, 8)
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
    offline_expected_qps = 2400000
    max_pairs_per_staging_thread = 262100
    start_from_device = False
    use_jemalloc = True
    scenario = Scenario.Offline
    benchmark = Benchmark.DLRM


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GBx8_HighAccuracy_Triton(BenchmarkConfiguration):
    system = System("A100-SXM-80GB", Architecture.Ampere, 8)
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
    offline_expected_qps = 2200000
    max_pairs_per_staging_thread = 262100
    start_from_device = False
    use_jemalloc = True
    scenario = Scenario.Offline
    benchmark = Benchmark.DLRM
    batch_triton_requests = True
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx8_Triton(BenchmarkConfiguration):
    system = System("A100-SXM-80GB", Architecture.Ampere, 8)
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
    gpu_batch_size = 262100
    offline_expected_qps = 2450000
    max_pairs_per_staging_thread = 262100
    start_from_device = False
    use_jemalloc = True
    scenario = Scenario.Offline
    benchmark = Benchmark.DLRM
    batch_triton_requests = True
    use_triton = True
