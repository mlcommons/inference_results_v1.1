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
    bert_opt_seqlen = 384
    coalesced_tensor = True
    enable_interleaved = False
    gpu_copy_streams = 1
    gpu_inference_streams = 2
    input_dtype = "int32"
    input_format = "linear"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_graphs = True
    active_sms = 60
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 48
    graphs_max_seqlen = 240
    server_num_issue_query_threads = 0
    server_target_qps = 12700
    soft_drop = 0.99
    start_from_device = False
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GBx4_HighAccuracy(BenchmarkConfiguration):
    system = System("A100-SXM-80GB", Architecture.Ampere, 4)
    bert_opt_seqlen = 384
    coalesced_tensor = True
    enable_interleaved = False
    gpu_copy_streams = 1
    gpu_inference_streams = 2
    input_dtype = "int32"
    input_format = "linear"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_graphs = True
    active_sms = 60
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 24
    graphs_max_seqlen = 240
    server_num_issue_query_threads = 0
    server_target_qps = 6000
    soft_drop = 0.99
    start_from_device = False
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GBx4_HighAccuracy_Triton(BenchmarkConfiguration):
    system = System("A100-SXM-80GB", Architecture.Ampere, 4)
    bert_opt_seqlen = 384
    coalesced_tensor = True
    enable_interleaved = False
    gpu_copy_streams = 1
    gpu_inference_streams = 2
    input_dtype = "int32"
    input_format = "linear"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_graphs = True
    active_sms = 60
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 64
    graphs_max_seqlen = 240
    server_num_issue_query_threads = 0
    server_target_qps = 5600
    soft_drop = 0.99
    start_from_device = False
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx4_Triton(BenchmarkConfiguration):
    system = System("A100-SXM-80GB", Architecture.Ampere, 4)
    bert_opt_seqlen = 384
    coalesced_tensor = True
    enable_interleaved = False
    gpu_copy_streams = 1
    gpu_inference_streams = 2
    input_dtype = "int32"
    input_format = "linear"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_graphs = True
    active_sms = 60
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 48
    graphs_max_seqlen = 240
    server_num_issue_query_threads = 0
    server_target_qps = 11500
    soft_drop = 0.99
    start_from_device = False
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx8(BenchmarkConfiguration):
    system = System("A100-SXM-80GB", Architecture.Ampere, 8)
    bert_opt_seqlen = 384
    coalesced_tensor = True
    enable_interleaved = False
    gpu_copy_streams = 1
    gpu_inference_streams = 2
    input_dtype = "int32"
    input_format = "linear"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_graphs = True
    active_sms = 60
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 48
    graphs_max_seqlen = 240
    server_num_issue_query_threads = 0
    server_target_qps = 25800
    soft_drop = 0.99
    start_from_device = False
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GBx8_HighAccuracy(BenchmarkConfiguration):
    system = System("A100-SXM-80GB", Architecture.Ampere, 8)
    bert_opt_seqlen = 384
    coalesced_tensor = True
    enable_interleaved = False
    gpu_copy_streams = 1
    gpu_inference_streams = 2
    input_dtype = "int32"
    input_format = "linear"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_graphs = True
    active_sms = 60
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 24
    graphs_max_seqlen = 240
    server_num_issue_query_threads = 0
    server_target_qps = 13100
    soft_drop = 0.99
    start_from_device = False
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GBx8_HighAccuracy_Triton(BenchmarkConfiguration):
    system = System("A100-SXM-80GB", Architecture.Ampere, 8)
    bert_opt_seqlen = 384
    coalesced_tensor = True
    enable_interleaved = False
    gpu_copy_streams = 1
    gpu_inference_streams = 2
    input_dtype = "int32"
    input_format = "linear"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_graphs = True
    active_sms = 60
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 64
    graphs_max_seqlen = 240
    server_num_issue_query_threads = 0
    server_target_qps = 11205
    soft_drop = 0.99
    start_from_device = False
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx8_Triton(BenchmarkConfiguration):
    system = System("A100-SXM-80GB", Architecture.Ampere, 8)
    bert_opt_seqlen = 384
    coalesced_tensor = True
    enable_interleaved = False
    gpu_copy_streams = 1
    gpu_inference_streams = 2
    input_dtype = "int32"
    input_format = "linear"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_graphs = True
    active_sms = 60
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 48
    graphs_max_seqlen = 240
    server_num_issue_query_threads = 0
    server_target_qps = 23000
    soft_drop = 0.99
    start_from_device = False
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT
    use_triton = True
