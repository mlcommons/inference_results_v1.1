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
    disable_encoder_plugin = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "fp16"
    input_format = "linear"
    map_path = "data_maps/rnnt_dev_clean_512/val_map.txt"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/rnnt_dev_clean_512/fp16"
    use_graphs = True
    audio_batch_size = 1
    audio_buffer_num_lines = 1
    audio_fp16_input = True
    dali_batches_issue_ahead = 1
    dali_pipeline_depth = 1
    single_stream_expected_latency_ns = 10000000
    nobatch_sorting = True
    nouse_copy_kernel = False
    num_warmups = 32
    scenario = Scenario.SingleStream
    benchmark = Benchmark.RNNT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx1(BenchmarkConfiguration):
    system = System("A100-PCIe-80GB", Architecture.Ampere, 1)
    disable_encoder_plugin = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "fp16"
    input_format = "linear"
    map_path = "data_maps/rnnt_dev_clean_512/val_map.txt"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/rnnt_dev_clean_512/fp16"
    use_graphs = True
    audio_batch_size = 1
    audio_buffer_num_lines = 1
    audio_fp16_input = True
    dali_batches_issue_ahead = 1
    dali_pipeline_depth = 1
    single_stream_expected_latency_ns = 10000000
    nobatch_sorting = True
    nouse_copy_kernel = False
    num_warmups = 32
    scenario = Scenario.SingleStream
    benchmark = Benchmark.RNNT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x1(BenchmarkConfiguration):
    system = System("A100-PCIe-80GB", Architecture.Ampere, 1, cpu_arch=CPUArch.aarch64)
    disable_encoder_plugin = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "fp16"
    input_format = "linear"
    map_path = "data_maps/rnnt_dev_clean_512/val_map.txt"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/rnnt_dev_clean_512/fp16"
    use_graphs = True
    audio_batch_size = 1
    audio_buffer_num_lines = 1
    audio_fp16_input = True
    dali_batches_issue_ahead = 1
    dali_pipeline_depth = 1
    single_stream_expected_latency_ns = 10000000
    nobatch_sorting = True
    nouse_copy_kernel = False
    num_warmups = 32
    scenario = Scenario.SingleStream
    benchmark = Benchmark.RNNT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_aarch64x1(BenchmarkConfiguration):
    system = System("A100-PCIe", Architecture.Ampere, 1, cpu_arch=CPUArch.aarch64)
    disable_encoder_plugin = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "fp16"
    input_format = "linear"
    map_path = "data_maps/rnnt_dev_clean_512/val_map.txt"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/rnnt_dev_clean_512/fp16"
    use_graphs = True
    audio_batch_size = 1
    audio_buffer_num_lines = 1
    audio_fp16_input = True
    dali_batches_issue_ahead = 1
    dali_pipeline_depth = 1
    single_stream_expected_latency_ns = 10000000
    nobatch_sorting = True
    nouse_copy_kernel = False
    num_warmups = 32
    scenario = Scenario.SingleStream
    benchmark = Benchmark.RNNT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb(BenchmarkConfiguration):
    _mig_configuration = MIGConfiguration({0: {MIGSlice(1, 10): 1}})
    system = System("A100-SXM-80GB", Architecture.Ampere, 1, mig_conf=_mig_configuration)
    disable_encoder_plugin = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "fp16"
    input_format = "linear"
    map_path = "data_maps/rnnt_dev_clean_512/val_map.txt"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/rnnt_dev_clean_512/fp16"
    use_graphs = True
    audio_batch_size = 1
    audio_buffer_num_lines = 4
    audio_fp16_input = True
    dali_batches_issue_ahead = 1
    dali_pipeline_depth = 1
    single_stream_expected_latency_ns = 10000000
    nobatch_sorting = True
    nouse_copy_kernel = True
    num_warmups = 32
    workspace_size = 1073741824
    scenario = Scenario.SingleStream
    benchmark = Benchmark.RNNT


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb_Hetero(A100_SXM_80GB_MIG_1x1g10gb):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx1(BenchmarkConfiguration):
    system = System("A100-SXM-80GB", Architecture.Ampere, 1)
    disable_encoder_plugin = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "fp16"
    input_format = "linear"
    map_path = "data_maps/rnnt_dev_clean_512/val_map.txt"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/rnnt_dev_clean_512/fp16"
    use_graphs = True
    audio_batch_size = 1
    audio_buffer_num_lines = 4
    audio_fp16_input = True
    dali_batches_issue_ahead = 1
    dali_pipeline_depth = 1
    single_stream_expected_latency_ns = 10000000
    nobatch_sorting = True
    nouse_copy_kernel = True
    num_warmups = 32
    start_from_device = True
    scenario = Scenario.SingleStream
    benchmark = Benchmark.RNNT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_40GB_MIG_1x1g5gb(BenchmarkConfiguration):
    _mig_configuration = MIGConfiguration({0: {MIGSlice(1, 5): 1}})
    system = System("A100-SXM4-40GB", Architecture.Ampere, 1, mig_conf=_mig_configuration)
    disable_encoder_plugin = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "fp16"
    input_format = "linear"
    map_path = "data_maps/rnnt_dev_clean_512/val_map.txt"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/rnnt_dev_clean_512/fp16"
    use_graphs = True
    audio_batch_size = 1
    audio_buffer_num_lines = 4
    audio_fp16_input = True
    dali_batches_issue_ahead = 1
    dali_pipeline_depth = 1
    single_stream_expected_latency_ns = 10000000
    nobatch_sorting = True
    nouse_copy_kernel = True
    num_warmups = 32
    workspace_size = 1073741824
    scenario = Scenario.SingleStream
    benchmark = Benchmark.RNNT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_40GBx1(BenchmarkConfiguration):
    system = System("A100-SXM4-40GB", Architecture.Ampere, 1)
    disable_encoder_plugin = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "fp16"
    input_format = "linear"
    map_path = "data_maps/rnnt_dev_clean_512/val_map.txt"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/rnnt_dev_clean_512/fp16"
    use_graphs = True
    audio_batch_size = 1
    audio_buffer_num_lines = 4
    audio_fp16_input = True
    dali_batches_issue_ahead = 1
    dali_pipeline_depth = 1
    single_stream_expected_latency_ns = 10000000
    nobatch_sorting = True
    nouse_copy_kernel = True
    num_warmups = 32
    start_from_device = True
    scenario = Scenario.SingleStream
    benchmark = Benchmark.RNNT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A10x1(BenchmarkConfiguration):
    system = System("A10", Architecture.Ampere, 1)
    disable_encoder_plugin = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "fp16"
    input_format = "linear"
    map_path = "data_maps/rnnt_dev_clean_512/val_map.txt"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/rnnt_dev_clean_512/fp16"
    use_graphs = True
    audio_batch_size = 1
    audio_buffer_num_lines = 1
    audio_fp16_input = True
    dali_batches_issue_ahead = 1
    dali_pipeline_depth = 1
    single_stream_expected_latency_ns = 25000000
    nobatch_sorting = True
    nouse_copy_kernel = True
    num_warmups = 32
    scenario = Scenario.SingleStream
    benchmark = Benchmark.RNNT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb(BenchmarkConfiguration):
    _mig_configuration = MIGConfiguration({0: {MIGSlice(1, 6): 1}})
    system = System("A30", Architecture.Ampere, 1, mig_conf=_mig_configuration)
    disable_encoder_plugin = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "fp16"
    input_format = "linear"
    map_path = "data_maps/rnnt_dev_clean_512/val_map.txt"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/rnnt_dev_clean_512/fp16"
    use_graphs = True
    audio_batch_size = 32
    audio_buffer_num_lines = 512
    audio_fp16_input = True
    dali_batches_issue_ahead = 1
    dali_pipeline_depth = 1
    single_stream_expected_latency_ns = 76133687
    nobatch_sorting = True
    nouse_copy_kernel = False
    num_warmups = 32
    workspace_size = 1610612736
    scenario = Scenario.SingleStream
    benchmark = Benchmark.RNNT


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_Hetero(A30_MIG_1x1g6gb):
    single_stream_expected_latency_ns = 78812921


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
    disable_encoder_plugin = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "fp16"
    input_format = "linear"
    map_path = "data_maps/rnnt_dev_clean_512/val_map.txt"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/rnnt_dev_clean_512/fp16"
    use_graphs = True
    audio_batch_size = 32
    audio_buffer_num_lines = 512
    audio_fp16_input = True
    dali_batches_issue_ahead = 1
    dali_pipeline_depth = 1
    single_stream_expected_latency_ns = 76133687
    nobatch_sorting = True
    nouse_copy_kernel = False
    num_warmups = 32
    workspace_size = 1610612736
    scenario = Scenario.SingleStream
    benchmark = Benchmark.RNNT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x1(BenchmarkConfiguration):
    system = System("A30", Architecture.Ampere, 1)
    disable_encoder_plugin = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "fp16"
    input_format = "linear"
    map_path = "data_maps/rnnt_dev_clean_512/val_map.txt"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/rnnt_dev_clean_512/fp16"
    use_graphs = True
    audio_batch_size = 1
    audio_buffer_num_lines = 1
    audio_fp16_input = True
    dali_batches_issue_ahead = 1
    dali_pipeline_depth = 1
    single_stream_expected_latency_ns = 10000000
    nobatch_sorting = True
    nouse_copy_kernel = False
    num_warmups = 32
    scenario = Scenario.SingleStream
    benchmark = Benchmark.RNNT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class AGX_Xavier(BenchmarkConfiguration):
    system = System("AGX_Xavier", Architecture.Xavier, 1, cpu_arch=CPUArch.aarch64)
    disable_encoder_plugin = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "fp16"
    input_format = "linear"
    map_path = "data_maps/rnnt_dev_clean_512/val_map.txt"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/rnnt_dev_clean_512/fp16"
    use_graphs = True
    audio_batch_size = 1
    single_stream_expected_latency_ns = 100000000
    scenario = Scenario.SingleStream
    benchmark = Benchmark.RNNT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class AGX_Xavier_MaxQ(BenchmarkConfiguration):
    system = System("AGX_Xavier", Architecture.Xavier, 1, cpu_arch=CPUArch.aarch64)
    disable_encoder_plugin = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "fp16"
    input_format = "linear"
    map_path = "data_maps/rnnt_dev_clean_512/val_map.txt"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/rnnt_dev_clean_512/fp16"
    use_graphs = True
    audio_batch_size = 1
    single_stream_expected_latency_ns = 100000000
    scenario = Scenario.SingleStream
    benchmark = Benchmark.RNNT

    # power settings
    xavier_gpu_freq = 1032750000
    xavier_dla_freq = 115200000
    xavier_cpu_freq = 1190400
    xavier_emc_freq = 1600000000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x1(BenchmarkConfiguration):
    system = System("T4", Architecture.Turing, 1)
    disable_encoder_plugin = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "fp16"
    input_format = "linear"
    map_path = "data_maps/rnnt_dev_clean_512/val_map.txt"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/rnnt_dev_clean_512/fp16"
    use_graphs = True
    audio_batch_size = 1
    audio_buffer_num_lines = 4
    audio_fp16_input = True
    dali_batches_issue_ahead = 1
    dali_pipeline_depth = 1
    single_stream_expected_latency_ns = 25000000
    nobatch_sorting = True
    nouse_copy_kernel = True
    num_warmups = 32
    scenario = Scenario.SingleStream
    benchmark = Benchmark.RNNT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class Xavier_NX(BenchmarkConfiguration):
    system = System("Xavier_NX", Architecture.Xavier, 1, cpu_arch=CPUArch.aarch64)
    disable_encoder_plugin = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "fp16"
    input_format = "linear"
    map_path = "data_maps/rnnt_dev_clean_512/val_map.txt"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/rnnt_dev_clean_512/fp16"
    use_graphs = True
    audio_batch_size = 1
    single_stream_expected_latency_ns = 200000000
    scenario = Scenario.SingleStream
    benchmark = Benchmark.RNNT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class Xavier_NX_MaxQ(BenchmarkConfiguration):
    system = System("Xavier_NX", Architecture.Xavier, 1, cpu_arch=CPUArch.aarch64)
    disable_encoder_plugin = True
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    input_dtype = "fp16"
    input_format = "linear"
    map_path = "data_maps/rnnt_dev_clean_512/val_map.txt"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/rnnt_dev_clean_512/fp16"
    use_graphs = True
    audio_batch_size = 1
    single_stream_expected_latency_ns = 200000000
    scenario = Scenario.SingleStream
    benchmark = Benchmark.RNNT

    # power settings
    xavier_gpu_freq = 752250000
    xavier_dla_freq = 115200000
    xavier_cpu_freq = 1190400
    xavier_emc_freq = 1331200000
