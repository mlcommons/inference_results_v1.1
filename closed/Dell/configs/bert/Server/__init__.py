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
class DSS8440_A100_PCIE_80GBx10(BenchmarkConfiguration):
    system = System("DSS8440_A100-PCIE-80GB", Architecture.Ampere, 10)
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
    gpu_batch_size = 64
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 30110
    soft_drop = 1.0
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT
    start_from_device = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class DSS8440_A100_PCIE_80GBx10_HighAccuracy(BenchmarkConfiguration):
    system = System("DSS8440_A100-PCIE-80GB", Architecture.Ampere, 10)
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
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 14540
    soft_drop = 1.0
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT
    start_from_device = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class DSS8440_A30x8(BenchmarkConfiguration):
    system = System("DSS8440_A30", Architecture.Ampere, 8)
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
    gpu_batch_size = 64
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 11500
    soft_drop = 0.993
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class DSS8440_A30x8_HighAccuracy(BenchmarkConfiguration):
    system = System("DSS8440_A30", Architecture.Ampere, 8)
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
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 5250
    soft_drop = 0.993
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class DSS8440_A30x8_HighAccuracy_Triton(BenchmarkConfiguration):
    system = System("DSS8440_A30", Architecture.Ampere, 8)
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
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 5200
    soft_drop = 0.993
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class DSS8440_A30x8_Triton(BenchmarkConfiguration):
    system = System("DSS8440_A30", Architecture.Ampere, 8)
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
    gpu_batch_size = 64
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 11000
    soft_drop = 0.993
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT
    use_triton = True
    max_queue_delay_usec = 1000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class R750xa_A100_PCIE_40GBx4_MaxQ(BenchmarkConfiguration):
    system = System("R750xa_A100-PCIE-40GB", Architecture.Ampere, 4)
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
    gpu_batch_size = 64
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 7800
    soft_drop = 1.0
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT
    start_from_device = True
    power_limit=175


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class R750xa_A100_PCIE_40GBx4_MaxQ_HighAccuracy(BenchmarkConfiguration):
    system = System("R750xa_A100-PCIE-40GB", Architecture.Ampere, 4)
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
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 3550
    soft_drop = 1.0
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT
    start_from_device = True
    power_limit=175


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750xa_A100_PCIE_80GBx4(BenchmarkConfiguration):
    system = System("R750xa_A100-PCIE-80GB", Architecture.Ampere, 4)
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
    gpu_batch_size = 64
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 11700
    soft_drop = 1.0
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT
    start_from_device = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R750xa_A100_PCIE_80GBx4_HighAccuracy(BenchmarkConfiguration):
    system = System("R750xa_A100-PCIE-80GB", Architecture.Ampere, 4)
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
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 5680
    soft_drop = 1.0
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT
    start_from_device = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750xa_A100_PCIE_80GBx4_Triton(R750xa_A100_PCIE_80GBx4):
    use_triton = True
    server_target_qps = 10700
    start_from_device = False


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R750xa_A100_PCIE_80GBx4_HighAccuracy_Triton(R750xa_A100_PCIE_80GBx4_HighAccuracy):
    use_triton = True
    start_from_device = False


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750xa_A100_PCIE_80GB_MIG_28x1g10gb_Triton(R750xa_A100_PCIE_80GBx4_Triton):
    _mig_configuration = MIGConfiguration({
        0: {MIGSlice(1, 10): 7},
        1: {MIGSlice(1, 10): 7},
        2: {MIGSlice(1, 10): 7},
        3: {MIGSlice(1, 10): 7},
    })
    system = System("R750xa_A100-PCIE-80GB", Architecture.Ampere, 4, mig_conf=_mig_configuration)
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    precision = "fp16"
    active_sms = 100
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 8
    server_num_issue_query_threads = 0
    server_target_qps = 9300
    soft_drop = 0.99
    deque_timeout_usec = 2000
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R750xa_A100_PCIE_80GB_MIG_28x1g10gb_HighAccuracy_Triton(R750xa_A100_PCIE_80GBx4_HighAccuracy_Triton):
    _mig_configuration = MIGConfiguration({
        0: {MIGSlice(1, 10): 7},
        1: {MIGSlice(1, 10): 7},
        2: {MIGSlice(1, 10): 7},
        3: {MIGSlice(1, 10): 7},
    })
    system = System("R750xa_A100-PCIE-80GB", Architecture.Ampere, 4, mig_conf=_mig_configuration)
    bert_opt_seqlen = 384
    coalesced_tensor = True
    enable_interleaved = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int32"
    input_format = "linear"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_graphs = True
    active_sms = 100
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 8
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 0
    server_target_qps = 10000
    soft_drop = 0.99
    use_small_tile_gemm_plugin = True
    deque_timeout_usec = 1000
    scenario = Scenario.Server
    benchmark = Benchmark.BERT
    use_triton = True
    start_from_device = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R7525_A100_PCIE_40GBx3(BenchmarkConfiguration):
    system = System("R7525_A100-PCIE-40GB", Architecture.Ampere, 3)
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
    active_sms = 152
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 69
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 7870
    soft_drop = 1.0
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT
    start_from_device=True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R7525_A100_PCIE_40GBx3_HighAccuracy(BenchmarkConfiguration):
    system = System("R7525_A100-PCIE-40GB", Architecture.Ampere, 3)
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
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 3812
    soft_drop = 1.0
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT
    start_from_device=True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R7525_A30x3(BenchmarkConfiguration):
    system = System("R7525_A30", Architecture.Ampere, 3)
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
    gpu_batch_size = 64
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 3960
    soft_drop = 0.993
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R7525_A30x3_HighAccuracy(BenchmarkConfiguration):
    system = System("R7525_A30", Architecture.Ampere, 3)
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
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 650*3
    soft_drop = 0.993
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE2420_A10x2(BenchmarkConfiguration):
    system = System("XE2420_A10", Architecture.Ampere, 2)
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
    active_sms = 100
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 16
    graphs_max_seqlen = 240
    server_num_issue_query_threads = 0
    server_target_qps = 1850
    soft_drop = 0.993
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT


# inherits XE2420_A10x2-99 values
@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class XE2420_A10x2_MaxQ(XE2420_A10x2):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE2420_A10x2_HighAccuracy(XE2420_A10x2):
    precision = "fp16"
    gpu_batch_size = 8
    server_target_qps = 840


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class XE2420_A10x2_HighAccuracy_MaxQ(XE2420_A10x2_HighAccuracy):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8545_A100_SXM_80GBx4(BenchmarkConfiguration):
    system = System("XE8545_A100-SXM-80GB", Architecture.Ampere, 4)
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
    server_target_qps = 13790
    start_from_device = True
    soft_drop = 0.99
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE8545_A100_SXM_80GBx4_HighAccuracy(BenchmarkConfiguration):
    system = System("XE8545_A100-SXM-80GB", Architecture.Ampere, 4)
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
    server_target_qps = 6900
    start_from_device = True
    soft_drop = 0.99
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8545_A100_SXM_80GBx4_Triton(BenchmarkConfiguration):
    system = System("XE8545_A100-SXM-80GB", Architecture.Ampere, 4)
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
    active_sms = 45
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 98 
    graphs_max_seqlen = 240
    server_num_issue_query_threads = 0
    server_target_qps = 12926 
    soft_drop = 0.99
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT
    use_triton = True

@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE8545_A100_SXM_80GBx4_HighAccuracy_Triton(BenchmarkConfiguration):
    system = System("XE8545_A100-SXM-80GB", Architecture.Ampere, 4)
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
    active_sms = 61
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 59
    graphs_max_seqlen = 240
    server_num_issue_query_threads = 0
    server_target_qps = 6400
    soft_drop = 0.99
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR12_A10x2(BenchmarkConfiguration):
    system = System("XR12_A10", Architecture.Ampere, 2)
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
    active_sms = 100
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 16
    graphs_max_seqlen = 240
    server_num_issue_query_threads = 0
    server_target_qps = 1850
    soft_drop = 0.993
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XR12_A10x2_HighAccuracy(XR12_A10x2):
    precision = "fp16"
    gpu_batch_size = 8
    server_target_qps = 840


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class vA100_PCIex3(BenchmarkConfiguration):
    system = System("R7525_vA100-PCIE-40GB", Architecture.Ampere, 3)
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
    gpu_batch_size = 64
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 7800.0
    soft_drop = 1.0
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class vA100_PCIex3_HighAccuracy(BenchmarkConfiguration):
    system = System("R7525_vA100-PCIE-40GB", Architecture.Ampere, 3)
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
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 3600
    soft_drop = 1.0
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT
