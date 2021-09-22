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
    server_target_qps = 2600
    soft_drop = 1.0
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIex1_HighAccuracy(BenchmarkConfiguration):
    system = System("A100-PCIe", Architecture.Ampere, 1)
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
    server_target_qps = 1215
    soft_drop = 1.0
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIex1_HighAccuracy_Triton(BenchmarkConfiguration):
    system = System("A100-PCIe", Architecture.Ampere, 1)
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
    server_target_qps = 1185
    soft_drop = 1.0
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIex1_Triton(BenchmarkConfiguration):
    system = System("A100-PCIe", Architecture.Ampere, 1)
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
    server_target_qps = 2450
    soft_drop = 1.0
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIex8(BenchmarkConfiguration):
    system = System("A100-PCIe", Architecture.Ampere, 8)
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
    server_target_qps = 20800.0
    soft_drop = 1.0
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIex8_HighAccuracy(BenchmarkConfiguration):
    system = System("A100-PCIe", Architecture.Ampere, 8)
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
    server_target_qps = 9600
    soft_drop = 1.0
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIex8_HighAccuracy_Triton(BenchmarkConfiguration):
    system = System("A100-PCIe", Architecture.Ampere, 8)
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
    server_target_qps = 9500
    soft_drop = 1.0
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIex8_Triton(BenchmarkConfiguration):
    system = System("A100-PCIe", Architecture.Ampere, 8)
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
    server_target_qps = 18000
    soft_drop = 1.0
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_PCIex8_MaxQ(BenchmarkConfiguration):
    system = System("A100-PCIe", Architecture.Ampere, 8)
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
    server_target_qps = 17500
    soft_drop = 1.0
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT
    power_limit = 200


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class A100_PCIex8_HighAccuracy_MaxQ(BenchmarkConfiguration):
    system = System("A100-PCIe", Architecture.Ampere, 8)
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
    server_target_qps = 7500
    soft_drop = 1.0
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT
    power_limit = 200


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class A100_PCIex8_HighAccuracy_Triton_MaxQ(BenchmarkConfiguration):
    system = System("A100-PCIe", Architecture.Ampere, 8)
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
    server_target_qps = 9480
    soft_drop = 1.0
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT
    power_limit = 200
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_PCIex8_Triton_MaxQ(BenchmarkConfiguration):
    system = System("A100-PCIe", Architecture.Ampere, 8)
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
    server_target_qps = 17000
    soft_drop = 1.0
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT
    power_limit = 200
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx8(BenchmarkConfiguration):
    system = System("A100-PCIe-80GB", Architecture.Ampere, 8)
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
    server_target_qps = 23000.0
    soft_drop = 1.0
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_80GBx8_HighAccuracy(A100_PCIe_80GBx8):
    precision = "fp16"
    server_target_qps = 10800


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx8_Triton(A100_PCIe_80GBx8):
    server_target_qps = 18000
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_80GBx8_HighAccuracy_Triton(A100_PCIe_80GBx8_HighAccuracy):
    server_target_qps = 9500
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x2(BenchmarkConfiguration):
    system = System("A100-PCIe-80GB", Architecture.Ampere, 2, cpu_arch=CPUArch.aarch64)
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
    server_target_qps = 5200.0
    soft_drop = 1.0
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x2_HighAccuracy(A100_PCIe_80GB_aarch64x2):
    precision = "fp16"
    server_target_qps = 2400


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x4(BenchmarkConfiguration):
    system = System("A100-PCIe-80GB", Architecture.Ampere, 4, cpu_arch=CPUArch.aarch64)
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
    server_target_qps = 10400.0
    soft_drop = 1.0
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x4_HighAccuracy(A100_PCIe_80GB_aarch64x4):
    precision = "fp16"
    server_target_qps = 4800


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_PCIe_80GB_aarch64x4_MaxQ(A100_PCIe_80GB_aarch64x4):
    server_target_qps = 10400.0
    power_limit = 225


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class A100_PCIe_80GB_aarch64x4_HighAccuracy_MaxQ(A100_PCIe_80GB_aarch64x4_MaxQ):
    precision = "fp16"
    server_target_qps = 4000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_aarch64x2(BenchmarkConfiguration):
    system = System("A100-PCIe", Architecture.Ampere, 2, cpu_arch=CPUArch.aarch64)
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
    server_target_qps = 5200.0
    soft_drop = 1.0
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_aarch64x2_HighAccuracy(A100_PCIe_aarch64x2):
    precision = "fp16"
    server_target_qps = 2400


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_aarch64x4(BenchmarkConfiguration):
    system = System("A100-PCIe", Architecture.Ampere, 4, cpu_arch=CPUArch.aarch64)
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
    server_target_qps = 10700.0
    soft_drop = 1.0
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_aarch64x4_HighAccuracy(A100_PCIe_aarch64x4):
    precision = "fp16"
    server_target_qps = 5050


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_PCIe_aarch64x4_MaxQ(A100_PCIe_aarch64x4):
    server_target_qps = 10100.0
    power_limit = 225


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class A100_PCIe_aarch64x4_HighAccuracy_MaxQ(A100_PCIe_aarch64x4_MaxQ):
    precision = "fp16"
    server_target_qps = 4000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb(BenchmarkConfiguration):
    _mig_configuration = MIGConfiguration({0: {MIGSlice(1, 10): 1}})
    system = System("A100-SXM-80GB", Architecture.Ampere, 1, mig_conf=_mig_configuration)
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
    gpu_batch_size = 16
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 0
    server_target_qps = 380
    soft_drop = 0.99
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb_Hetero(A100_SXM_80GB_MIG_1x1g10gb):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb_HighAccuracy(BenchmarkConfiguration):
    _mig_configuration = MIGConfiguration({0: {MIGSlice(1, 10): 1}})
    system = System("A100-SXM-80GB", Architecture.Ampere, 1, mig_conf=_mig_configuration)
    bert_opt_seqlen = 384
    coalesced_tensor = True
    enable_interleaved = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int32"
    input_format = "linear"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_graphs = True
    active_sms = 100
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 16
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 0
    server_target_qps = 170
    soft_drop = 0.99
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb_Hetero_HighAccuracy(A100_SXM_80GB_MIG_1x1g10gb_HighAccuracy):
    server_target_qps = 160


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb_HighAccuracy_Triton(BenchmarkConfiguration):
    _mig_configuration = MIGConfiguration({0: {MIGSlice(1, 10): 1}})
    system = System("A100-SXM-80GB", Architecture.Ampere, 1, mig_conf=_mig_configuration)
    bert_opt_seqlen = 384
    coalesced_tensor = True
    enable_interleaved = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int32"
    input_format = "linear"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_graphs = True
    active_sms = 100
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 16
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 0
    server_target_qps = 170
    soft_drop = 0.99
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb_Triton(BenchmarkConfiguration):
    _mig_configuration = MIGConfiguration({0: {MIGSlice(1, 10): 1}})
    system = System("A100-SXM-80GB", Architecture.Ampere, 1, mig_conf=_mig_configuration)
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
    gpu_batch_size = 16
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 0
    server_target_qps = 360
    soft_drop = 0.99
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
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
    gpu_batch_size = 12
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 0
    server_target_qps = 20160
    soft_drop = 0.99
    use_small_tile_gemm_plugin = True
    deque_timeout_usec = 1000
    scenario = Scenario.Server
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_56x1g10gb_HighAccuracy(BenchmarkConfiguration):
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
    bert_opt_seqlen = 384
    coalesced_tensor = True
    enable_interleaved = False
    gpu_copy_streams = 2
    gpu_inference_streams = 2
    input_dtype = "int32"
    input_format = "linear"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_graphs = True
    active_sms = 100
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 16
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 0
    server_target_qps = 170
    soft_drop = 0.99
    use_small_tile_gemm_plugin = True
    deque_timeout_usec = 1000
    scenario = Scenario.Server
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_56x1g10gb_HighAccuracy_Triton(BenchmarkConfiguration):
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
    bert_opt_seqlen = 384
    coalesced_tensor = True
    enable_interleaved = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int32"
    input_format = "linear"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_graphs = True
    active_sms = 100
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 8
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 0
    server_target_qps = 9300
    soft_drop = 0.99
    use_small_tile_gemm_plugin = True
    deque_timeout_usec = 2000
    scenario = Scenario.Server
    benchmark = Benchmark.BERT
    use_triton = True


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
    gpu_batch_size = 12
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 0
    server_target_qps = 20500
    soft_drop = 0.99
    use_small_tile_gemm_plugin = True
    deque_timeout_usec = 1000
    scenario = Scenario.Server
    benchmark = Benchmark.BERT
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx1(BenchmarkConfiguration):
    system = System("A100-SXM-80GB", Architecture.Ampere, 1)
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
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 3200
    soft_drop = 0.99
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GBx1_HighAccuracy(BenchmarkConfiguration):
    system = System("A100-SXM-80GB", Architecture.Ampere, 1)
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
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 1550
    soft_drop = 0.99
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GBx1_HighAccuracy_Triton(BenchmarkConfiguration):
    system = System("A100-SXM-80GB", Architecture.Ampere, 1)
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
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 1400
    soft_drop = 0.99
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx1_Triton(BenchmarkConfiguration):
    system = System("A100-SXM-80GB", Architecture.Ampere, 1)
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
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 2800
    soft_drop = 0.99
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx4(BenchmarkConfiguration):
    _system_alias = "DGX Station A100 - Red October"
    _notes = "This should not inherit from A100_SXM_80GB (DGX-A100), and cannot use start_from_device"

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
    gpu_batch_size = 64
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 10800
    soft_drop = 1.0
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GBx4_HighAccuracy(BenchmarkConfiguration):
    _system_alias = "DGX Station A100 - Red October"
    _notes = "This should not inherit from A100_SXM_80GB (DGX-A100), and cannot use start_from_device"

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
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 4860
    soft_drop = 1.0
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GBx4_HighAccuracy_Triton(BenchmarkConfiguration):
    _system_alias = "DGX Station A100 - Red October"
    _notes = "This should not inherit from A100_SXM_80GB (DGX-A100), and cannot use start_from_device"

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
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 4650
    soft_drop = 1.0
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx4_Triton(BenchmarkConfiguration):
    _system_alias = "DGX Station A100 - Red October"
    _notes = "This should not inherit from A100_SXM_80GB (DGX-A100), and cannot use start_from_device"

    system = System("A100-SXM-80GB", Architecture.Ampere, 4)
    bert_opt_seqlen = 384
    coalesced_tensor = True
    enable_interleaved = False
    gpu_copy_streams = 1
    instance_group_count = 2
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
    server_target_qps = 10800
    soft_drop = 1.0
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_SXM_80GBx4_MaxQ(BenchmarkConfiguration):
    _system_alias = "DGX Station A100 - Red October"
    _notes = "This should not inherit from A100_SXM_80GB (DGX-A100), and cannot use start_from_device"

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
    gpu_batch_size = 64
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 10200
    soft_drop = 1.0
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT
    power_limit = 250


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class A100_SXM_80GBx4_HighAccuracy_MaxQ(BenchmarkConfiguration):
    _system_alias = "DGX Station A100 - Red October"
    _notes = "This should not inherit from A100_SXM_80GB (DGX-A100), and cannot use start_from_device"

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
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 4300
    soft_drop = 1.0
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT
    power_limit = 250


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class A100_SXM_80GBx4_HighAccuracy_Triton_MaxQ(BenchmarkConfiguration):
    _system_alias = "DGX Station A100 - Red October"
    _notes = "This should not inherit from A100_SXM_80GB (DGX-A100), and cannot use start_from_device"

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
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 4500
    soft_drop = 1.0
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT
    power_limit = 250
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_SXM_80GBx4_Triton_MaxQ(BenchmarkConfiguration):
    _system_alias = "DGX Station A100 - Red October"
    _notes = "This should not inherit from A100_SXM_80GB (DGX-A100), and cannot use start_from_device"

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
    gpu_batch_size = 64
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 8780
    soft_drop = 1.0
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT
    power_limit = 250
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
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_SXM_80GBx8_MaxQ(BenchmarkConfiguration):
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
    server_target_qps = 21500
    soft_drop = 0.99
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT
    power_limit = 275


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class A100_SXM_80GBx8_HighAccuracy_MaxQ(BenchmarkConfiguration):
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
    server_target_qps = 10000
    soft_drop = 0.99
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT
    power_limit = 275


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class A100_SXM_80GBx8_HighAccuracy_Triton_MaxQ(BenchmarkConfiguration):
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
    gpu_batch_size = 48
    graphs_max_seqlen = 240
    server_num_issue_query_threads = 0
    server_target_qps = 11205
    soft_drop = 0.99
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT
    power_limit = 275
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_SXM_80GBx8_Triton_MaxQ(BenchmarkConfiguration):
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
    server_target_qps = 22455
    soft_drop = 0.99
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT
    power_limit = 275
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_40GBx1(BenchmarkConfiguration):
    system = System("A100-SXM4-40GB", Architecture.Ampere, 1)
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
    server_target_qps = 3100
    soft_drop = 0.99
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM4_40GBx1_HighAccuracy(BenchmarkConfiguration):
    system = System("A100-SXM4-40GB", Architecture.Ampere, 1)
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
    server_target_qps = 1550
    soft_drop = 0.99
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM4_40GBx1_HighAccuracy_Triton(BenchmarkConfiguration):
    system = System("A100-SXM4-40GB", Architecture.Ampere, 1)
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
    server_target_qps = 1400
    soft_drop = 0.99
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_40GBx1_Triton(BenchmarkConfiguration):
    system = System("A100-SXM4-40GB", Architecture.Ampere, 1)
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
    server_target_qps = 2800
    soft_drop = 0.99
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_40GBx8(BenchmarkConfiguration):
    system = System("A100-SXM4-40GB", Architecture.Ampere, 8)
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
    gpu_batch_size = 96
    graphs_max_seqlen = 240
    server_num_issue_query_threads = 0
    server_target_qps = 24750
    soft_drop = 0.99
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM4_40GBx8_HighAccuracy(BenchmarkConfiguration):
    system = System("A100-SXM4-40GB", Architecture.Ampere, 8)
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
    server_target_qps = 11500
    soft_drop = 0.99
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM4_40GBx8_HighAccuracy_Triton(BenchmarkConfiguration):
    system = System("A100-SXM4-40GB", Architecture.Ampere, 8)
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
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_40GBx8_Triton(BenchmarkConfiguration):
    system = System("A100-SXM4-40GB", Architecture.Ampere, 8)
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
    gpu_batch_size = 96
    graphs_max_seqlen = 240
    server_num_issue_query_threads = 0
    server_target_qps = 22455
    soft_drop = 0.99
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A10x1(BenchmarkConfiguration):
    system = System("A10", Architecture.Ampere, 1)
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
    server_target_qps = 900
    soft_drop = 0.993
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A10x1_HighAccuracy(BenchmarkConfiguration):
    system = System("A10", Architecture.Ampere, 1)
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
    active_sms = 100
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 8
    graphs_max_seqlen = 240
    server_num_issue_query_threads = 0
    server_target_qps = 390
    soft_drop = 0.993
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A10x1_HighAccuracy_Triton(BenchmarkConfiguration):
    system = System("A10", Architecture.Ampere, 1)
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
    active_sms = 100
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 8
    graphs_max_seqlen = 240
    server_num_issue_query_threads = 0
    server_target_qps = 360
    soft_drop = 0.993
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A10x1_Triton(BenchmarkConfiguration):
    system = System("A10", Architecture.Ampere, 1)
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
    server_target_qps = 800
    soft_drop = 0.993
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A10x8(BenchmarkConfiguration):
    system = System("A10", Architecture.Ampere, 8)
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
    server_target_qps = 7200.0
    soft_drop = 0.993
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A10x8_HighAccuracy(BenchmarkConfiguration):
    system = System("A10", Architecture.Ampere, 8)
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
    active_sms = 100
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 8
    graphs_max_seqlen = 240
    server_num_issue_query_threads = 0
    server_target_qps = 3120.0
    soft_drop = 0.993
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A10x8_HighAccuracy_Triton(BenchmarkConfiguration):
    system = System("A10", Architecture.Ampere, 8)
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
    active_sms = 100
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 8
    graphs_max_seqlen = 240
    server_num_issue_query_threads = 0
    server_target_qps = 3200
    soft_drop = 0.993
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A10x8_Triton(BenchmarkConfiguration):
    system = System("A10", Architecture.Ampere, 8)
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
    server_target_qps = 7000
    soft_drop = 0.993
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT
    use_triton = True
    max_queue_delay_usec = 9000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb(BenchmarkConfiguration):
    _mig_configuration = MIGConfiguration({0: {MIGSlice(1, 6): 1}})
    system = System("A30", Architecture.Ampere, 1, mig_conf=_mig_configuration)
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
    active_sms = 60
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 10
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 0
    server_target_qps = 380
    soft_drop = 0.99
    use_small_tile_gemm_plugin = True
    deque_timeout_usec = 50000
    workspace_size = 805306368
    scenario = Scenario.Server
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_HighAccuracy(BenchmarkConfiguration):
    _mig_configuration = MIGConfiguration({0: {MIGSlice(1, 6): 1}})
    system = System("A30", Architecture.Ampere, 1, mig_conf=_mig_configuration)
    bert_opt_seqlen = 384
    coalesced_tensor = True
    enable_interleaved = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int32"
    input_format = "linear"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_graphs = True
    active_sms = 60
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 6
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 0
    server_target_qps = 150
    soft_drop = 0.99
    use_small_tile_gemm_plugin = True
    deque_timeout_usec = 50000
    workspace_size = 805306368
    scenario = Scenario.Server
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_Hetero(A30_MIG_1x1g6gb):
    server_target_qps = 340


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_Hetero_HighAccuracy(A30_MIG_1x1g6gb_HighAccuracy):
    server_target_qps = 130


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_HighAccuracy_Triton(BenchmarkConfiguration):
    _mig_configuration = MIGConfiguration({0: {MIGSlice(1, 6): 1}})
    system = System("A30", Architecture.Ampere, 1, mig_conf=_mig_configuration)
    bert_opt_seqlen = 384
    coalesced_tensor = True
    enable_interleaved = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int32"
    input_format = "linear"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_graphs = True
    active_sms = 60
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 64
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 0
    server_target_qps = 120
    soft_drop = 0.99
    use_small_tile_gemm_plugin = True
    deque_timeout_usec = 50000
    workspace_size = 805306368
    scenario = Scenario.Server
    benchmark = Benchmark.BERT
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_Triton(BenchmarkConfiguration):
    _mig_configuration = MIGConfiguration({0: {MIGSlice(1, 6): 1}})
    system = System("A30", Architecture.Ampere, 1, mig_conf=_mig_configuration)
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
    active_sms = 60
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 10
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 0
    server_target_qps = 260
    soft_drop = 0.99
    use_small_tile_gemm_plugin = True
    deque_timeout_usec = 50000
    workspace_size = 805306368
    scenario = Scenario.Server
    benchmark = Benchmark.BERT
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
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
    active_sms = 60
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 10
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 0
    server_target_qps = 10000
    soft_drop = 0.99
    use_small_tile_gemm_plugin = True
    deque_timeout_usec = 1000
    workspace_size = 805306368
    scenario = Scenario.Server
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30_MIG_32x1g6gb_HighAccuracy(BenchmarkConfiguration):
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
    bert_opt_seqlen = 384
    coalesced_tensor = True
    enable_interleaved = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int32"
    input_format = "linear"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_graphs = True
    active_sms = 60
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 6
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 0
    server_target_qps = 150
    soft_drop = 0.99
    use_small_tile_gemm_plugin = True
    deque_timeout_usec = 1000
    workspace_size = 805306368
    scenario = Scenario.Server
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30_MIG_32x1g6gb_HighAccuracy_Triton(BenchmarkConfiguration):
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
    bert_opt_seqlen = 384
    coalesced_tensor = True
    enable_interleaved = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int32"
    input_format = "linear"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_graphs = True
    active_sms = 60
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 6
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 0
    server_target_qps = 3800
    soft_drop = 0.99
    use_small_tile_gemm_plugin = True
    deque_timeout_usec = 1000
    workspace_size = 805306368
    scenario = Scenario.Server
    benchmark = Benchmark.BERT
    use_triton = True


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
    active_sms = 60
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 10
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 0
    server_target_qps = 8300
    soft_drop = 0.99
    use_small_tile_gemm_plugin = True
    deque_timeout_usec = 1000
    workspace_size = 805306368
    scenario = Scenario.Server
    benchmark = Benchmark.BERT
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x1(BenchmarkConfiguration):
    system = System("A30", Architecture.Ampere, 1)
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
    server_target_qps = 1500
    soft_drop = 0.993
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30x1_HighAccuracy(BenchmarkConfiguration):
    system = System("A30", Architecture.Ampere, 1)
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
    server_target_qps = 650
    soft_drop = 0.993
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30x1_HighAccuracy_Triton(BenchmarkConfiguration):
    system = System("A30", Architecture.Ampere, 1)
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
    server_target_qps = 600
    soft_drop = 0.993
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x1_Triton(BenchmarkConfiguration):
    system = System("A30", Architecture.Ampere, 1)
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
    server_target_qps = 1400
    soft_drop = 0.993
    use_small_tile_gemm_plugin = True
    scenario = Scenario.Server
    benchmark = Benchmark.BERT
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x8(BenchmarkConfiguration):
    system = System("A30", Architecture.Ampere, 8)
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
class A30x8_HighAccuracy(BenchmarkConfiguration):
    system = System("A30", Architecture.Ampere, 8)
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
class A30x8_HighAccuracy_Triton(BenchmarkConfiguration):
    system = System("A30", Architecture.Ampere, 8)
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
class A30x8_Triton(BenchmarkConfiguration):
    system = System("A30", Architecture.Ampere, 8)
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


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x1(BenchmarkConfiguration):
    system = System("T4", Architecture.Turing, 1)
    bert_opt_seqlen = 384
    coalesced_tensor = True
    enable_interleaved = True
    gpu_copy_streams = 1
    gpu_inference_streams = 2
    input_dtype = "int32"
    input_format = "linear"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_graphs = True
    active_sms = 100
    gpu_batch_size = 16
    graphs_max_seqlen = 240
    server_num_issue_query_threads = 0
    server_target_qps = 360
    soft_drop = 0.993
    scenario = Scenario.Server
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class T4x1_HighAccuracy(BenchmarkConfiguration):
    system = System("T4", Architecture.Turing, 1)
    bert_opt_seqlen = 384
    coalesced_tensor = True
    enable_interleaved = True
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int32"
    input_format = "linear"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_graphs = True
    active_sms = 100
    gpu_batch_size = 8
    graphs_max_seqlen = 240
    server_num_issue_query_threads = 0
    server_target_qps = 160
    soft_drop = 0.993
    scenario = Scenario.Server
    benchmark = Benchmark.BERT
    graph_specs = "(128, 4, 256, 4), (192, 128, 512, 4), (256, 192, 1536, 8), (384, 256, 2048, 16)"


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class T4x1_HighAccuracy_Triton(BenchmarkConfiguration):
    system = System("T4", Architecture.Turing, 1)
    bert_opt_seqlen = 384
    coalesced_tensor = True
    enable_interleaved = True
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int32"
    input_format = "linear"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_graphs = True
    active_sms = 100
    gpu_batch_size = 8
    graphs_max_seqlen = 240
    server_num_issue_query_threads = 0
    server_target_qps = 144
    soft_drop = 0.993
    scenario = Scenario.Server
    benchmark = Benchmark.BERT
    graph_specs = "(128, 4, 256, 4), (192, 128, 512, 4), (256, 192, 1536, 8), (384, 256, 2048, 16)"
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x1_Triton(BenchmarkConfiguration):
    system = System("T4", Architecture.Turing, 1)
    bert_opt_seqlen = 384
    coalesced_tensor = True
    enable_interleaved = True
    gpu_copy_streams = 1
    gpu_inference_streams = 2
    input_dtype = "int32"
    input_format = "linear"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_graphs = True
    active_sms = 100
    gpu_batch_size = 16
    graphs_max_seqlen = 240
    server_num_issue_query_threads = 0
    server_target_qps = 324
    soft_drop = 0.993
    scenario = Scenario.Server
    benchmark = Benchmark.BERT
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x20(BenchmarkConfiguration):
    system = System("T4", Architecture.Turing, 20)
    bert_opt_seqlen = 384
    coalesced_tensor = True
    enable_interleaved = True
    gpu_copy_streams = 1
    gpu_inference_streams = 2
    input_dtype = "int32"
    input_format = "linear"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_graphs = True
    active_sms = 100
    gpu_batch_size = 14
    graphs_max_seqlen = 260
    server_num_issue_query_threads = 40
    server_target_qps = 5000
    soft_drop = 0.995
    scenario = Scenario.Server
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class T4x20_HighAccuracy(BenchmarkConfiguration):
    system = System("T4", Architecture.Turing, 20)
    bert_opt_seqlen = 384
    coalesced_tensor = True
    enable_interleaved = True
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int32"
    input_format = "linear"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_graphs = True
    active_sms = 100
    gpu_batch_size = 8
    graphs_max_seqlen = 260
    server_num_issue_query_threads = 20
    server_target_qps = 3300
    soft_drop = 0.992
    scenario = Scenario.Server
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class T4x20_HighAccuracy_Triton(BenchmarkConfiguration):
    system = System("T4", Architecture.Turing, 20)
    bert_opt_seqlen = 384
    coalesced_tensor = True
    enable_interleaved = True
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int32"
    input_format = "linear"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_graphs = True
    active_sms = 100
    gpu_batch_size = 8
    graphs_max_seqlen = 260
    server_num_issue_query_threads = 20
    server_target_qps = 3300
    soft_drop = 0.992
    scenario = Scenario.Server
    benchmark = Benchmark.BERT
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x20_Triton(BenchmarkConfiguration):
    system = System("T4", Architecture.Turing, 20)
    bert_opt_seqlen = 384
    coalesced_tensor = True
    enable_interleaved = True
    gpu_copy_streams = 1
    gpu_inference_streams = 2
    input_dtype = "int32"
    input_format = "linear"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_graphs = True
    active_sms = 100
    gpu_batch_size = 14
    graphs_max_seqlen = 260
    server_num_issue_query_threads = 40
    server_target_qps = 5000
    soft_drop = 0.995
    scenario = Scenario.Server
    benchmark = Benchmark.BERT
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x8(BenchmarkConfiguration):
    system = System("T4", Architecture.Turing, 8)
    bert_opt_seqlen = 384
    coalesced_tensor = True
    enable_interleaved = True
    gpu_copy_streams = 1
    gpu_inference_streams = 2
    input_dtype = "int32"
    input_format = "linear"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_graphs = True
    active_sms = 100
    gpu_batch_size = 14
    graphs_max_seqlen = 260
    server_num_issue_query_threads = 16
    server_target_qps = 2200
    soft_drop = 0.992
    scenario = Scenario.Server
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class T4x8_HighAccuracy(BenchmarkConfiguration):
    system = System("T4", Architecture.Turing, 8)
    bert_opt_seqlen = 384
    coalesced_tensor = True
    enable_interleaved = True
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int32"
    input_format = "linear"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_graphs = True
    active_sms = 100
    gpu_batch_size = 8
    graphs_max_seqlen = 260
    server_num_issue_query_threads = 8
    server_target_qps = 1330
    soft_drop = 0.992
    scenario = Scenario.Server
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class T4x8_HighAccuracy_Triton(BenchmarkConfiguration):
    system = System("T4", Architecture.Turing, 8)
    bert_opt_seqlen = 384
    coalesced_tensor = True
    enable_interleaved = True
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int32"
    input_format = "linear"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_graphs = True
    active_sms = 100
    gpu_batch_size = 8
    graphs_max_seqlen = 260
    server_num_issue_query_threads = 8
    server_target_qps = 1330
    soft_drop = 0.992
    scenario = Scenario.Server
    benchmark = Benchmark.BERT
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x8_Triton(BenchmarkConfiguration):
    system = System("T4", Architecture.Turing, 8)
    bert_opt_seqlen = 384
    coalesced_tensor = True
    enable_interleaved = True
    gpu_copy_streams = 1
    gpu_inference_streams = 2
    input_dtype = "int32"
    input_format = "linear"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_graphs = True
    active_sms = 100
    gpu_batch_size = 14
    graphs_max_seqlen = 260
    server_num_issue_query_threads = 16
    server_target_qps = 2200
    soft_drop = 0.992
    scenario = Scenario.Server
    benchmark = Benchmark.BERT
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class Triton_CPU_2S_6258Rx1_Triton(BenchmarkConfiguration):
    system = System("Triton_CPU_2S_6258R", Architecture.Intel_CPU_x86_64, 1)
    bert_opt_seqlen = 384
    coalesced_tensor = True
    input_dtype = "fp32"
    precision = "fp32"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy"
    batch_size = 0
    server_target_qps = 1
    scenario = Scenario.Server
    benchmark = Benchmark.BERT
    model_name = "bert_int8_openvino"
    num_instances = 26
    ov_parameters = {'CPU_THROUGHPUT_STREAMS': '14', 'SKIP_OV_DYNAMIC_BATCHSIZE': 'YES'}
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class Triton_CPU_4S_8380Hx1_Triton(BenchmarkConfiguration):
    system = System("Triton_CPU_4S_8380H", Architecture.Intel_CPU_x86_64, 1)
    bert_opt_seqlen = 384
    coalesced_tensor = True
    input_dtype = "fp32"
    precision = "fp32"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy"
    batch_size = 0
    server_target_qps = 26.5
    scenario = Scenario.Server
    benchmark = Benchmark.BERT
    model_name = "bert_int8_openvino"
    num_instances = 8
    ov_parameters = {'CPU_THREADS_NUM': '112', 'CPU_THROUGHPUT_STREAMS': '4', 'ENABLE_BATCH_PADDING': 'NO', 'SKIP_OV_DYNAMIC_BATCHSIZE': 'YES'}
    use_triton = True
