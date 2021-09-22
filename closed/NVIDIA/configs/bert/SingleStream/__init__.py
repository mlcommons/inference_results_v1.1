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
    bert_opt_seqlen = 270
    coalesced_tensor = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int32"
    input_format = "linear"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_small_tile_gemm_plugin = False
    enable_interleaved = False
    gemm_plugin_fairshare_cache_size = 120
    single_stream_expected_latency_ns = 1700000
    use_graphs = True
    scenario = Scenario.SingleStream
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIex1_HighAccuracy(BenchmarkConfiguration):
    system = System("A100-PCIe", Architecture.Ampere, 1)
    bert_opt_seqlen = 270
    coalesced_tensor = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int32"
    input_format = "linear"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_small_tile_gemm_plugin = False
    enable_interleaved = False
    gemm_plugin_fairshare_cache_size = 120
    single_stream_expected_latency_ns = 1700000
    use_graphs = True
    scenario = Scenario.SingleStream
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIex1_HighAccuracy_Triton(BenchmarkConfiguration):
    system = System("A100-PCIe", Architecture.Ampere, 1)
    bert_opt_seqlen = 270
    coalesced_tensor = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int32"
    input_format = "linear"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_small_tile_gemm_plugin = False
    enable_interleaved = False
    gemm_plugin_fairshare_cache_size = 120
    single_stream_expected_latency_ns = 1700000
    use_graphs = True
    scenario = Scenario.SingleStream
    benchmark = Benchmark.BERT
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIex1_Triton(BenchmarkConfiguration):
    system = System("A100-PCIe", Architecture.Ampere, 1)
    bert_opt_seqlen = 270
    coalesced_tensor = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int32"
    input_format = "linear"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_small_tile_gemm_plugin = False
    enable_interleaved = False
    gemm_plugin_fairshare_cache_size = 120
    single_stream_expected_latency_ns = 1700000
    use_graphs = True
    scenario = Scenario.SingleStream
    benchmark = Benchmark.BERT
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx1(BenchmarkConfiguration):
    system = System("A100-PCIe-80GB", Architecture.Ampere, 1)
    bert_opt_seqlen = 270
    coalesced_tensor = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int32"
    input_format = "linear"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_small_tile_gemm_plugin = False
    enable_interleaved = False
    gemm_plugin_fairshare_cache_size = 120
    single_stream_expected_latency_ns = 1700000
    use_graphs = True
    scenario = Scenario.SingleStream
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_80GBx1_HighAccuracy(A100_PCIe_80GBx1):
    precision = "fp16"
    single_stream_expected_latency_ns = 1700000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx1_Triton(A100_PCIe_80GBx1):
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_80GBx1_HighAccuracy_Triton(A100_PCIe_80GBx1_HighAccuracy):
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x1(BenchmarkConfiguration):
    system = System("A100-PCIe-80GB", Architecture.Ampere, 1, cpu_arch=CPUArch.aarch64)
    bert_opt_seqlen = 270
    coalesced_tensor = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int32"
    input_format = "linear"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_small_tile_gemm_plugin = False
    enable_interleaved = False
    gemm_plugin_fairshare_cache_size = 120
    single_stream_expected_latency_ns = 1700000
    use_graphs = True
    scenario = Scenario.SingleStream
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x1_HighAccuracy(A100_PCIe_80GB_aarch64x1):
    precision = "fp16"
    single_stream_expected_latency_ns = 1700000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_aarch64x1(BenchmarkConfiguration):
    system = System("A100-PCIe", Architecture.Ampere, 1, cpu_arch=CPUArch.aarch64)
    bert_opt_seqlen = 270
    coalesced_tensor = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int32"
    input_format = "linear"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_small_tile_gemm_plugin = False
    enable_interleaved = False
    gemm_plugin_fairshare_cache_size = 120
    single_stream_expected_latency_ns = 1700000
    use_graphs = True
    scenario = Scenario.SingleStream
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_aarch64x1_HighAccuracy(A100_PCIe_aarch64x1):
    precision = "fp16"
    single_stream_expected_latency_ns = 1700000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb(BenchmarkConfiguration):
    _mig_configuration = MIGConfiguration({0: {MIGSlice(1, 10): 1}})
    system = System("A100-SXM-80GB", Architecture.Ampere, 1, mig_conf=_mig_configuration)
    bert_opt_seqlen = 270
    coalesced_tensor = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int32"
    input_format = "linear"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_small_tile_gemm_plugin = False
    single_stream_expected_latency_ns = 1700000
    use_graphs = True
    workspace_size = 2147483648
    scenario = Scenario.SingleStream
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb_HighAccuracy(BenchmarkConfiguration):
    _mig_configuration = MIGConfiguration({0: {MIGSlice(1, 10): 1}})
    system = System("A100-SXM-80GB", Architecture.Ampere, 1, mig_conf=_mig_configuration)
    bert_opt_seqlen = 270
    coalesced_tensor = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int32"
    input_format = "linear"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_small_tile_gemm_plugin = False
    single_stream_expected_latency_ns = 1700000
    use_graphs = True
    workspace_size = 2147483648
    scenario = Scenario.SingleStream
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb_Hetero(A100_SXM_80GB_MIG_1x1g10gb):
    pass


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb_Hetero_HighAccuracy(A100_SXM_80GB_MIG_1x1g10gb_HighAccuracy):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb_HighAccuracy_Triton(BenchmarkConfiguration):
    _mig_configuration = MIGConfiguration({0: {MIGSlice(1, 10): 1}})
    system = System("A100-SXM-80GB", Architecture.Ampere, 1, mig_conf=_mig_configuration)
    bert_opt_seqlen = 270
    coalesced_tensor = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int32"
    input_format = "linear"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_small_tile_gemm_plugin = False
    single_stream_expected_latency_ns = 1700000
    use_graphs = True
    workspace_size = 2147483648
    scenario = Scenario.SingleStream
    benchmark = Benchmark.BERT
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb_Triton(BenchmarkConfiguration):
    _mig_configuration = MIGConfiguration({0: {MIGSlice(1, 10): 1}})
    system = System("A100-SXM-80GB", Architecture.Ampere, 1, mig_conf=_mig_configuration)
    bert_opt_seqlen = 270
    coalesced_tensor = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int32"
    input_format = "linear"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_small_tile_gemm_plugin = False
    single_stream_expected_latency_ns = 1700000
    use_graphs = True
    workspace_size = 2147483648
    scenario = Scenario.SingleStream
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
    bert_opt_seqlen = 270
    coalesced_tensor = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int32"
    input_format = "linear"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_small_tile_gemm_plugin = False
    single_stream_expected_latency_ns = 1700000
    use_graphs = True
    workspace_size = 2147483648
    scenario = Scenario.SingleStream
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
    bert_opt_seqlen = 270
    coalesced_tensor = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int32"
    input_format = "linear"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_small_tile_gemm_plugin = False
    single_stream_expected_latency_ns = 1700000
    use_graphs = True
    workspace_size = 2147483648
    scenario = Scenario.SingleStream
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
    bert_opt_seqlen = 270
    coalesced_tensor = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int32"
    input_format = "linear"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_small_tile_gemm_plugin = False
    single_stream_expected_latency_ns = 1700000
    use_graphs = True
    workspace_size = 2147483648
    scenario = Scenario.SingleStream
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
    bert_opt_seqlen = 270
    coalesced_tensor = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int32"
    input_format = "linear"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_small_tile_gemm_plugin = False
    single_stream_expected_latency_ns = 1700000
    use_graphs = True
    workspace_size = 2147483648
    scenario = Scenario.SingleStream
    benchmark = Benchmark.BERT
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx1(BenchmarkConfiguration):
    system = System("A100-SXM-80GB", Architecture.Ampere, 1)
    bert_opt_seqlen = 270
    coalesced_tensor = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int32"
    input_format = "linear"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_small_tile_gemm_plugin = False
    enable_interleaved = False
    gemm_plugin_fairshare_cache_size = 120
    single_stream_expected_latency_ns = 1700000
    use_graphs = True
    scenario = Scenario.SingleStream
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GBx1_HighAccuracy(BenchmarkConfiguration):
    system = System("A100-SXM-80GB", Architecture.Ampere, 1)
    bert_opt_seqlen = 270
    coalesced_tensor = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int32"
    input_format = "linear"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_small_tile_gemm_plugin = False
    enable_interleaved = False
    gemm_plugin_fairshare_cache_size = 120
    single_stream_expected_latency_ns = 1700000
    use_graphs = True
    scenario = Scenario.SingleStream
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GBx1_HighAccuracy_Triton(BenchmarkConfiguration):
    system = System("A100-SXM-80GB", Architecture.Ampere, 1)
    bert_opt_seqlen = 270
    coalesced_tensor = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int32"
    input_format = "linear"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_small_tile_gemm_plugin = False
    enable_interleaved = False
    gemm_plugin_fairshare_cache_size = 120
    single_stream_expected_latency_ns = 1700000
    use_graphs = True
    scenario = Scenario.SingleStream
    benchmark = Benchmark.BERT
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx1_Triton(BenchmarkConfiguration):
    system = System("A100-SXM-80GB", Architecture.Ampere, 1)
    bert_opt_seqlen = 270
    coalesced_tensor = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int32"
    input_format = "linear"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_small_tile_gemm_plugin = False
    enable_interleaved = False
    gemm_plugin_fairshare_cache_size = 120
    single_stream_expected_latency_ns = 1700000
    use_graphs = True
    scenario = Scenario.SingleStream
    benchmark = Benchmark.BERT
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_40GB_MIG_1x1g5gb(BenchmarkConfiguration):
    _mig_configuration = MIGConfiguration({0: {MIGSlice(1, 5): 1}})
    system = System("A100-SXM4-40GB", Architecture.Ampere, 1, mig_conf=_mig_configuration)
    bert_opt_seqlen = 270
    coalesced_tensor = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int32"
    input_format = "linear"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_small_tile_gemm_plugin = False
    single_stream_expected_latency_ns = 1700000
    use_graphs = True
    workspace_size = 2147483648
    scenario = Scenario.SingleStream
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM4_40GB_MIG_1x1g5gb_HighAccuracy(BenchmarkConfiguration):
    _mig_configuration = MIGConfiguration({0: {MIGSlice(1, 5): 1}})
    system = System("A100-SXM4-40GB", Architecture.Ampere, 1, mig_conf=_mig_configuration)
    bert_opt_seqlen = 270
    coalesced_tensor = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int32"
    input_format = "linear"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_small_tile_gemm_plugin = False
    single_stream_expected_latency_ns = 1700000
    use_graphs = True
    workspace_size = 2147483648
    scenario = Scenario.SingleStream
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM4_40GB_MIG_1x1g5gb_HighAccuracy_Triton(BenchmarkConfiguration):
    _mig_configuration = MIGConfiguration({0: {MIGSlice(1, 5): 1}})
    system = System("A100-SXM4-40GB", Architecture.Ampere, 1, mig_conf=_mig_configuration)
    bert_opt_seqlen = 270
    coalesced_tensor = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int32"
    input_format = "linear"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_small_tile_gemm_plugin = False
    single_stream_expected_latency_ns = 1700000
    use_graphs = True
    workspace_size = 2147483648
    scenario = Scenario.SingleStream
    benchmark = Benchmark.BERT
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_40GB_MIG_1x1g5gb_Triton(BenchmarkConfiguration):
    _mig_configuration = MIGConfiguration({0: {MIGSlice(1, 5): 1}})
    system = System("A100-SXM4-40GB", Architecture.Ampere, 1, mig_conf=_mig_configuration)
    bert_opt_seqlen = 270
    coalesced_tensor = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int32"
    input_format = "linear"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_small_tile_gemm_plugin = False
    single_stream_expected_latency_ns = 1700000
    use_graphs = True
    workspace_size = 2147483648
    scenario = Scenario.SingleStream
    benchmark = Benchmark.BERT
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_40GBx1(BenchmarkConfiguration):
    system = System("A100-SXM4-40GB", Architecture.Ampere, 1)
    bert_opt_seqlen = 270
    coalesced_tensor = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int32"
    input_format = "linear"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_small_tile_gemm_plugin = False
    enable_interleaved = False
    gemm_plugin_fairshare_cache_size = 120
    single_stream_expected_latency_ns = 1700000
    use_graphs = True
    scenario = Scenario.SingleStream
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM4_40GBx1_HighAccuracy(BenchmarkConfiguration):
    system = System("A100-SXM4-40GB", Architecture.Ampere, 1)
    bert_opt_seqlen = 270
    coalesced_tensor = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int32"
    input_format = "linear"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_small_tile_gemm_plugin = False
    enable_interleaved = False
    gemm_plugin_fairshare_cache_size = 120
    single_stream_expected_latency_ns = 1700000
    use_graphs = True
    scenario = Scenario.SingleStream
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM4_40GBx1_HighAccuracy_Triton(BenchmarkConfiguration):
    system = System("A100-SXM4-40GB", Architecture.Ampere, 1)
    bert_opt_seqlen = 270
    coalesced_tensor = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int32"
    input_format = "linear"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_small_tile_gemm_plugin = False
    enable_interleaved = False
    gemm_plugin_fairshare_cache_size = 120
    single_stream_expected_latency_ns = 1700000
    use_graphs = True
    scenario = Scenario.SingleStream
    benchmark = Benchmark.BERT
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_40GBx1_Triton(BenchmarkConfiguration):
    system = System("A100-SXM4-40GB", Architecture.Ampere, 1)
    bert_opt_seqlen = 270
    coalesced_tensor = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int32"
    input_format = "linear"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_small_tile_gemm_plugin = False
    enable_interleaved = False
    gemm_plugin_fairshare_cache_size = 120
    single_stream_expected_latency_ns = 1700000
    use_graphs = True
    scenario = Scenario.SingleStream
    benchmark = Benchmark.BERT
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A10x1(BenchmarkConfiguration):
    system = System("A10", Architecture.Ampere, 1)
    bert_opt_seqlen = 270
    coalesced_tensor = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int32"
    input_format = "linear"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_small_tile_gemm_plugin = False
    enable_interleaved = False
    gemm_plugin_fairshare_cache_size = 120
    single_stream_expected_latency_ns = 2800000
    use_graphs = True
    scenario = Scenario.SingleStream
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A10x1_HighAccuracy(BenchmarkConfiguration):
    system = System("A10", Architecture.Ampere, 1)
    bert_opt_seqlen = 270
    coalesced_tensor = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int32"
    input_format = "linear"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_small_tile_gemm_plugin = False
    enable_interleaved = False
    gemm_plugin_fairshare_cache_size = 120
    single_stream_expected_latency_ns = 2800000
    use_graphs = True
    scenario = Scenario.SingleStream
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A10x1_HighAccuracy_Triton(BenchmarkConfiguration):
    system = System("A10", Architecture.Ampere, 1)
    bert_opt_seqlen = 270
    coalesced_tensor = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int32"
    input_format = "linear"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_small_tile_gemm_plugin = False
    enable_interleaved = False
    gemm_plugin_fairshare_cache_size = 120
    single_stream_expected_latency_ns = 2800000
    use_graphs = True
    scenario = Scenario.SingleStream
    benchmark = Benchmark.BERT
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A10x1_Triton(BenchmarkConfiguration):
    system = System("A10", Architecture.Ampere, 1)
    bert_opt_seqlen = 270
    coalesced_tensor = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int32"
    input_format = "linear"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_small_tile_gemm_plugin = False
    enable_interleaved = False
    gemm_plugin_fairshare_cache_size = 120
    single_stream_expected_latency_ns = 2800000
    use_graphs = True
    scenario = Scenario.SingleStream
    benchmark = Benchmark.BERT
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb(BenchmarkConfiguration):
    _mig_configuration = MIGConfiguration({0: {MIGSlice(1, 6): 1}})
    system = System("A30", Architecture.Ampere, 1, mig_conf=_mig_configuration)
    bert_opt_seqlen = 270
    coalesced_tensor = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int32"
    input_format = "linear"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_small_tile_gemm_plugin = False
    enable_interleaved = False
    gemm_plugin_fairshare_cache_size = 120
    single_stream_expected_latency_ns = 5999404
    use_graphs = True
    workspace_size = 1610612736
    scenario = Scenario.SingleStream
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_HighAccuracy(BenchmarkConfiguration):
    _mig_configuration = MIGConfiguration({0: {MIGSlice(1, 6): 1}})
    system = System("A30", Architecture.Ampere, 1, mig_conf=_mig_configuration)
    bert_opt_seqlen = 270
    coalesced_tensor = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int32"
    input_format = "linear"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_small_tile_gemm_plugin = False
    enable_interleaved = False
    gemm_plugin_fairshare_cache_size = 120
    single_stream_expected_latency_ns = 11000950
    use_graphs = True
    workspace_size = 1610612736
    scenario = Scenario.SingleStream
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_Hetero(A30_MIG_1x1g6gb):
    single_stream_expected_latency_ns = 6412655


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_Hetero_HighAccuracy(A30_MIG_1x1g6gb_HighAccuracy):
    single_stream_expected_latency_ns = 11725740


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_HighAccuracy_Triton(BenchmarkConfiguration):
    _mig_configuration = MIGConfiguration({0: {MIGSlice(1, 6): 1}})
    system = System("A30", Architecture.Ampere, 1, mig_conf=_mig_configuration)
    bert_opt_seqlen = 270
    coalesced_tensor = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int32"
    input_format = "linear"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_small_tile_gemm_plugin = False
    enable_interleaved = False
    gemm_plugin_fairshare_cache_size = 120
    single_stream_expected_latency_ns = 7452826
    use_graphs = True
    workspace_size = 1610612736
    scenario = Scenario.SingleStream
    benchmark = Benchmark.BERT
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_Triton(BenchmarkConfiguration):
    _mig_configuration = MIGConfiguration({0: {MIGSlice(1, 6): 1}})
    system = System("A30", Architecture.Ampere, 1, mig_conf=_mig_configuration)
    bert_opt_seqlen = 270
    coalesced_tensor = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int32"
    input_format = "linear"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_small_tile_gemm_plugin = False
    enable_interleaved = False
    gemm_plugin_fairshare_cache_size = 120
    single_stream_expected_latency_ns = 5999404
    use_graphs = True
    workspace_size = 1610612736
    scenario = Scenario.SingleStream
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
    bert_opt_seqlen = 270
    coalesced_tensor = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int32"
    input_format = "linear"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_small_tile_gemm_plugin = False
    enable_interleaved = False
    gemm_plugin_fairshare_cache_size = 120
    single_stream_expected_latency_ns = 7452826
    use_graphs = True
    workspace_size = 1610612736
    scenario = Scenario.SingleStream
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
    bert_opt_seqlen = 270
    coalesced_tensor = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int32"
    input_format = "linear"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_small_tile_gemm_plugin = False
    enable_interleaved = False
    gemm_plugin_fairshare_cache_size = 120
    single_stream_expected_latency_ns = 7452826
    use_graphs = True
    workspace_size = 1610612736
    scenario = Scenario.SingleStream
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
    bert_opt_seqlen = 270
    coalesced_tensor = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int32"
    input_format = "linear"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_small_tile_gemm_plugin = False
    enable_interleaved = False
    gemm_plugin_fairshare_cache_size = 120
    single_stream_expected_latency_ns = 7452826
    use_graphs = True
    workspace_size = 1610612736
    scenario = Scenario.SingleStream
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
    bert_opt_seqlen = 270
    coalesced_tensor = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int32"
    input_format = "linear"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_small_tile_gemm_plugin = False
    enable_interleaved = False
    gemm_plugin_fairshare_cache_size = 120
    single_stream_expected_latency_ns = 9400000
    use_graphs = True
    workspace_size = 1610612736
    scenario = Scenario.SingleStream
    benchmark = Benchmark.BERT
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x1(BenchmarkConfiguration):
    system = System("A30", Architecture.Ampere, 1)
    bert_opt_seqlen = 270
    coalesced_tensor = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int32"
    input_format = "linear"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_small_tile_gemm_plugin = False
    enable_interleaved = False
    gemm_plugin_fairshare_cache_size = 120
    single_stream_expected_latency_ns = 1700000
    use_graphs = True
    scenario = Scenario.SingleStream
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30x1_HighAccuracy(BenchmarkConfiguration):
    system = System("A30", Architecture.Ampere, 1)
    bert_opt_seqlen = 270
    coalesced_tensor = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int32"
    input_format = "linear"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_small_tile_gemm_plugin = False
    enable_interleaved = False
    gemm_plugin_fairshare_cache_size = 120
    single_stream_expected_latency_ns = 1700000
    use_graphs = True
    scenario = Scenario.SingleStream
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30x1_HighAccuracy_Triton(BenchmarkConfiguration):
    system = System("A30", Architecture.Ampere, 1)
    bert_opt_seqlen = 270
    coalesced_tensor = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int32"
    input_format = "linear"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_small_tile_gemm_plugin = False
    enable_interleaved = False
    gemm_plugin_fairshare_cache_size = 120
    single_stream_expected_latency_ns = 1700000
    use_graphs = True
    scenario = Scenario.SingleStream
    benchmark = Benchmark.BERT
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x1_Triton(BenchmarkConfiguration):
    system = System("A30", Architecture.Ampere, 1)
    bert_opt_seqlen = 270
    coalesced_tensor = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int32"
    input_format = "linear"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_small_tile_gemm_plugin = False
    enable_interleaved = False
    gemm_plugin_fairshare_cache_size = 120
    single_stream_expected_latency_ns = 3400000
    use_graphs = True
    scenario = Scenario.SingleStream
    benchmark = Benchmark.BERT
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class AGX_Xavier(BenchmarkConfiguration):
    system = System("AGX_Xavier", Architecture.Xavier, 1, cpu_arch=CPUArch.aarch64)
    bert_opt_seqlen = 270
    coalesced_tensor = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int32"
    input_format = "linear"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_small_tile_gemm_plugin = False
    single_stream_expected_latency_ns = 31000000
    use_graphs = False
    scenario = Scenario.SingleStream
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class AGX_Xavier_MaxQ(BenchmarkConfiguration):
    system = System("AGX_Xavier", Architecture.Xavier, 1, cpu_arch=CPUArch.aarch64)
    bert_opt_seqlen = 270
    coalesced_tensor = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int32"
    input_format = "linear"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_small_tile_gemm_plugin = False
    single_stream_expected_latency_ns = 31000000
    use_graphs = False
    scenario = Scenario.SingleStream
    benchmark = Benchmark.BERT

    # power settings
    xavier_gpu_freq = 828750000
    xavier_dla_freq = 115200000
    xavier_cpu_freq = 1190400
    xavier_emc_freq = 1600000000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class AGX_Xavier_HighAccuracy_MaxQ(BenchmarkConfiguration):
    system = System("AGX_Xavier", Architecture.Xavier, 1, cpu_arch=CPUArch.aarch64)
    bert_opt_seqlen = 270
    coalesced_tensor = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int32"
    input_format = "linear"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_small_tile_gemm_plugin = False
    single_stream_expected_latency_ns = 31000000
    use_graphs = False
    scenario = Scenario.SingleStream
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class AGX_Xavier_Triton(BenchmarkConfiguration):
    system = System("AGX_Xavier", Architecture.Xavier, 1, cpu_arch=CPUArch.aarch64)
    bert_opt_seqlen = 270
    coalesced_tensor = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int32"
    input_format = "linear"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_small_tile_gemm_plugin = False
    single_stream_expected_latency_ns = 31000000
    use_graphs = False
    scenario = Scenario.SingleStream
    benchmark = Benchmark.BERT
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x1(BenchmarkConfiguration):
    system = System("T4", Architecture.Turing, 1)
    bert_opt_seqlen = 270
    coalesced_tensor = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int32"
    input_format = "linear"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_small_tile_gemm_plugin = False
    single_stream_expected_latency_ns = 6400000
    use_graphs = True
    scenario = Scenario.SingleStream
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class T4x1_HighAccuracy(BenchmarkConfiguration):
    system = System("T4", Architecture.Turing, 1)
    bert_opt_seqlen = 270
    coalesced_tensor = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int32"
    input_format = "linear"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_small_tile_gemm_plugin = False
    single_stream_expected_latency_ns = 6400000
    use_graphs = True
    scenario = Scenario.SingleStream
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class T4x1_HighAccuracy_Triton(BenchmarkConfiguration):
    system = System("T4", Architecture.Turing, 1)
    bert_opt_seqlen = 270
    coalesced_tensor = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int32"
    input_format = "linear"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_small_tile_gemm_plugin = False
    single_stream_expected_latency_ns = 6400000
    use_graphs = True
    scenario = Scenario.SingleStream
    benchmark = Benchmark.BERT
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x1_Triton(BenchmarkConfiguration):
    system = System("T4", Architecture.Turing, 1)
    bert_opt_seqlen = 270
    coalesced_tensor = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int32"
    input_format = "linear"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_small_tile_gemm_plugin = False
    single_stream_expected_latency_ns = 6400000
    use_graphs = True
    scenario = Scenario.SingleStream
    benchmark = Benchmark.BERT
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class Xavier_NX(BenchmarkConfiguration):
    system = System("Xavier_NX", Architecture.Xavier, 1, cpu_arch=CPUArch.aarch64)
    bert_opt_seqlen = 270
    coalesced_tensor = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int32"
    input_format = "linear"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_small_tile_gemm_plugin = False
    single_stream_expected_latency_ns = 50000000
    use_graphs = False
    workspace_size = 1073741824
    scenario = Scenario.SingleStream
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class Xavier_NX_MaxQ(BenchmarkConfiguration):
    system = System("Xavier_NX", Architecture.Xavier, 1, cpu_arch=CPUArch.aarch64)
    bert_opt_seqlen = 270
    coalesced_tensor = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int32"
    input_format = "linear"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_small_tile_gemm_plugin = False
    single_stream_expected_latency_ns = 50000000
    use_graphs = False
    workspace_size = 1073741824
    scenario = Scenario.SingleStream
    benchmark = Benchmark.BERT

    # power settings
    xavier_gpu_freq = 854250000
    xavier_dla_freq = 115200000
    xavier_cpu_freq = 1190400
    xavier_emc_freq = 1600000000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class Xavier_NX_HighAccuracy_MaxQ(BenchmarkConfiguration):
    system = System("Xavier_NX", Architecture.Xavier, 1, cpu_arch=CPUArch.aarch64)
    bert_opt_seqlen = 270
    coalesced_tensor = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int32"
    input_format = "linear"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_small_tile_gemm_plugin = False
    single_stream_expected_latency_ns = 50000000
    use_graphs = False
    workspace_size = 1073741824
    scenario = Scenario.SingleStream
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class Xavier_NX_Triton(BenchmarkConfiguration):
    system = System("Xavier_NX", Architecture.Xavier, 1, cpu_arch=CPUArch.aarch64)
    bert_opt_seqlen = 270
    coalesced_tensor = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "int32"
    input_format = "linear"
    precision = "int8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/squad_tokenized/input_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/segment_ids.npy,${PREPROCESSED_DATA_DIR}/squad_tokenized/input_mask.npy"
    use_small_tile_gemm_plugin = False
    single_stream_expected_latency_ns = 50000000
    use_graphs = False
    workspace_size = 1073741824
    scenario = Scenario.SingleStream
    benchmark = Benchmark.BERT
    use_triton = True
