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
    gpu_inference_streams = 1
    input_dtype = "fp16"
    input_format = "linear"
    map_path = "data_maps/rnnt_dev_clean_512/val_map.txt"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/rnnt_dev_clean_512/fp16"
    use_graphs = True
    audio_batch_size = 1024
    audio_buffer_num_lines = 4096
    audio_fp16_input = True
    dali_batches_issue_ahead = 0
    dali_pipeline_depth = 2
    gpu_batch_size = 2048
    gpu_copy_streams = 1
    nobatch_sorting = True
    num_warmups = 20480
    server_target_qps = 11100
    scenario = Scenario.Server
    benchmark = Benchmark.RNNT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIex8(BenchmarkConfiguration):
    system = System("A100-PCIe", Architecture.Ampere, 8)
    gpu_inference_streams = 1
    input_dtype = "fp16"
    input_format = "linear"
    map_path = "data_maps/rnnt_dev_clean_512/val_map.txt"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/rnnt_dev_clean_512/fp16"
    use_graphs = True
    audio_batch_size = 1024
    audio_buffer_num_lines = 4096
    audio_fp16_input = True
    dali_batches_issue_ahead = 0
    dali_pipeline_depth = 2
    gpu_batch_size = 2048
    gpu_copy_streams = 1
    nobatch_sorting = True
    num_warmups = 20480
    server_target_qps = 86000
    scenario = Scenario.Server
    benchmark = Benchmark.RNNT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_PCIex8_MaxQ(BenchmarkConfiguration):
    system = System("A100-PCIe", Architecture.Ampere, 8)
    gpu_inference_streams = 1
    input_dtype = "fp16"
    input_format = "linear"
    map_path = "data_maps/rnnt_dev_clean_512/val_map.txt"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/rnnt_dev_clean_512/fp16"
    use_graphs = True
    audio_batch_size = 1024
    audio_buffer_num_lines = 4096
    audio_fp16_input = True
    dali_batches_issue_ahead = 0
    dali_pipeline_depth = 2
    gpu_batch_size = 2048
    gpu_copy_streams = 1
    nobatch_sorting = True
    num_warmups = 20480
    server_target_qps = 75000
    scenario = Scenario.Server
    benchmark = Benchmark.RNNT
    power_limit = 200


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx1(BenchmarkConfiguration):
    system = System("A100-PCIe-80GB", Architecture.Ampere, 1)
    gpu_inference_streams = 1
    input_dtype = "fp16"
    input_format = "linear"
    map_path = "data_maps/rnnt_dev_clean_512/val_map.txt"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/rnnt_dev_clean_512/fp16"
    use_graphs = True
    audio_batch_size = 1024
    audio_buffer_num_lines = 4096
    audio_fp16_input = True
    dali_batches_issue_ahead = 0
    dali_pipeline_depth = 2
    gpu_batch_size = 2048
    gpu_copy_streams = 1
    nobatch_sorting = True
    num_warmups = 20480
    server_target_qps = 11100
    scenario = Scenario.Server
    benchmark = Benchmark.RNNT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx8(BenchmarkConfiguration):
    system = System("A100-PCIe-80GB", Architecture.Ampere, 8)
    gpu_inference_streams = 1
    input_dtype = "fp16"
    input_format = "linear"
    map_path = "data_maps/rnnt_dev_clean_512/val_map.txt"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/rnnt_dev_clean_512/fp16"
    use_graphs = True
    audio_batch_size = 1024
    audio_buffer_num_lines = 4096
    audio_fp16_input = True
    dali_batches_issue_ahead = 0
    dali_pipeline_depth = 2
    gpu_batch_size = 2048
    gpu_copy_streams = 1
    nobatch_sorting = True
    num_warmups = 20480
    server_target_qps = 90000
    scenario = Scenario.Server
    benchmark = Benchmark.RNNT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x1(BenchmarkConfiguration):
    system = System("A100-PCIe-80GB", Architecture.Ampere, 1, cpu_arch=CPUArch.aarch64)
    gpu_inference_streams = 1
    input_dtype = "fp16"
    input_format = "linear"
    map_path = "data_maps/rnnt_dev_clean_512/val_map.txt"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/rnnt_dev_clean_512/fp16"
    use_graphs = True
    audio_batch_size = 1024
    audio_buffer_num_lines = 4096
    audio_fp16_input = True
    dali_batches_issue_ahead = 0
    dali_pipeline_depth = 2
    gpu_batch_size = 2048
    gpu_copy_streams = 1
    nobatch_sorting = True
    num_warmups = 20480
    server_target_qps = 11100
    scenario = Scenario.Server
    benchmark = Benchmark.RNNT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x2(BenchmarkConfiguration):
    system = System("A100-PCIe-80GB", Architecture.Ampere, 2, cpu_arch=CPUArch.aarch64)
    gpu_inference_streams = 1
    input_dtype = "fp16"
    input_format = "linear"
    map_path = "data_maps/rnnt_dev_clean_512/val_map.txt"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/rnnt_dev_clean_512/fp16"
    use_graphs = True
    audio_batch_size = 1024
    audio_buffer_num_lines = 4096
    audio_fp16_input = True
    dali_batches_issue_ahead = 0
    dali_pipeline_depth = 2
    gpu_batch_size = 2048
    gpu_copy_streams = 1
    nobatch_sorting = True
    num_warmups = 20480
    server_target_qps = 21500
    scenario = Scenario.Server
    benchmark = Benchmark.RNNT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x4(BenchmarkConfiguration):
    system = System("A100-PCIe-80GB", Architecture.Ampere, 4, cpu_arch=CPUArch.aarch64)
    gpu_inference_streams = 1
    input_dtype = "fp16"
    input_format = "linear"
    map_path = "data_maps/rnnt_dev_clean_512/val_map.txt"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/rnnt_dev_clean_512/fp16"
    use_graphs = True
    audio_batch_size = 1024
    audio_buffer_num_lines = 4096
    audio_fp16_input = True
    dali_batches_issue_ahead = 0
    dali_pipeline_depth = 2
    gpu_batch_size = 2048
    gpu_copy_streams = 1
    nobatch_sorting = True
    num_warmups = 20480
    server_target_qps = 43000
    scenario = Scenario.Server
    benchmark = Benchmark.RNNT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_PCIe_80GB_aarch64x4_MaxQ(A100_PCIe_80GB_aarch64x4):
    server_target_qps = 37500
    power_limit = 200


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_aarch64x1(BenchmarkConfiguration):
    system = System("A100-PCIe", Architecture.Ampere, 1, cpu_arch=CPUArch.aarch64)
    gpu_inference_streams = 1
    input_dtype = "fp16"
    input_format = "linear"
    map_path = "data_maps/rnnt_dev_clean_512/val_map.txt"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/rnnt_dev_clean_512/fp16"
    use_graphs = True
    audio_batch_size = 1024
    audio_buffer_num_lines = 4096
    audio_fp16_input = True
    dali_batches_issue_ahead = 0
    dali_pipeline_depth = 2
    gpu_batch_size = 2048
    gpu_copy_streams = 1
    nobatch_sorting = True
    num_warmups = 20480
    server_target_qps = 11100
    scenario = Scenario.Server
    benchmark = Benchmark.RNNT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_aarch64x2(BenchmarkConfiguration):
    system = System("A100-PCIe", Architecture.Ampere, 2, cpu_arch=CPUArch.aarch64)
    gpu_inference_streams = 1
    input_dtype = "fp16"
    input_format = "linear"
    map_path = "data_maps/rnnt_dev_clean_512/val_map.txt"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/rnnt_dev_clean_512/fp16"
    use_graphs = True
    audio_batch_size = 1024
    audio_buffer_num_lines = 4096
    audio_fp16_input = True
    dali_batches_issue_ahead = 0
    dali_pipeline_depth = 2
    gpu_batch_size = 2048
    gpu_copy_streams = 1
    nobatch_sorting = True
    num_warmups = 20480
    server_target_qps = 21500
    scenario = Scenario.Server
    benchmark = Benchmark.RNNT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_aarch64x4(BenchmarkConfiguration):
    system = System("A100-PCIe", Architecture.Ampere, 4, cpu_arch=CPUArch.aarch64)
    gpu_inference_streams = 1
    input_dtype = "fp16"
    input_format = "linear"
    map_path = "data_maps/rnnt_dev_clean_512/val_map.txt"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/rnnt_dev_clean_512/fp16"
    use_graphs = True
    audio_batch_size = 1024
    audio_buffer_num_lines = 4096
    audio_fp16_input = True
    dali_batches_issue_ahead = 0
    dali_pipeline_depth = 1
    gpu_batch_size = 2048
    gpu_copy_streams = 1
    nobatch_sorting = True
    num_warmups = 20480
    server_target_qps = 42500
    scenario = Scenario.Server
    benchmark = Benchmark.RNNT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_PCIe_aarch64x4_MaxQ(A100_PCIe_aarch64x4):
    dali_pipeline_depth = 1
    server_target_qps = 38500
    power_limit = 200


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb(BenchmarkConfiguration):
    _mig_configuration = MIGConfiguration({0: {MIGSlice(1, 10): 1}})
    system = System("A100-SXM-80GB", Architecture.Ampere, 1, mig_conf=_mig_configuration)
    gpu_inference_streams = 1
    input_dtype = "fp16"
    input_format = "linear"
    map_path = "data_maps/rnnt_dev_clean_512/val_map.txt"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/rnnt_dev_clean_512/fp16"
    use_graphs = True
    audio_batch_size = 64
    audio_buffer_num_lines = 512
    audio_fp16_input = True
    dali_batches_issue_ahead = 2
    dali_pipeline_depth = 2
    gpu_batch_size = 1024
    gpu_copy_streams = 1
    nobatch_sorting = True
    num_warmups = 64
    server_target_qps = 1300
    start_from_device = True
    max_seq_length = 64
    scenario = Scenario.Server
    benchmark = Benchmark.RNNT


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb_Hetero(A100_SXM_80GB_MIG_1x1g10gb):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx1(BenchmarkConfiguration):
    system = System("A100-SXM-80GB", Architecture.Ampere, 1)
    gpu_inference_streams = 1
    input_dtype = "fp16"
    input_format = "linear"
    map_path = "data_maps/rnnt_dev_clean_512/val_map.txt"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/rnnt_dev_clean_512/fp16"
    use_graphs = True
    audio_batch_size = 1024
    audio_buffer_num_lines = 4096
    audio_fp16_input = True
    dali_batches_issue_ahead = 0
    dali_pipeline_depth = 2
    gpu_batch_size = 1792
    gpu_copy_streams = 1
    nobatch_sorting = True
    num_warmups = 20480
    server_target_qps = 12750
    start_from_device = True
    scenario = Scenario.Server
    benchmark = Benchmark.RNNT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx4(BenchmarkConfiguration):
    _system_alias = "DGX Station A100 - Red October"
    _notes = "This should not inherit from A100_SXM_80GB (DGX-A100), and cannot use start_from_device"

    system = System("A100-SXM-80GB", Architecture.Ampere, 4)
    gpu_inference_streams = 1
    input_dtype = "fp16"
    input_format = "linear"
    map_path = "data_maps/rnnt_dev_clean_512/val_map.txt"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/rnnt_dev_clean_512/fp16"
    use_graphs = True
    audio_batch_size = 1024
    audio_buffer_num_lines = 4096
    audio_fp16_input = True
    dali_batches_issue_ahead = 0
    dali_pipeline_depth = 2
    gpu_batch_size = 1792
    gpu_copy_streams = 1
    nobatch_sorting = True
    num_warmups = 20480
    server_target_qps = 45000
    scenario = Scenario.Server
    benchmark = Benchmark.RNNT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_SXM_80GBx4_MaxQ(BenchmarkConfiguration):
    _system_alias = "DGX Station A100 - Red October"
    _notes = "This should not inherit from A100_SXM_80GB (DGX-A100), and cannot use start_from_device"

    system = System("A100-SXM-80GB", Architecture.Ampere, 4)
    gpu_inference_streams = 1
    input_dtype = "fp16"
    input_format = "linear"
    map_path = "data_maps/rnnt_dev_clean_512/val_map.txt"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/rnnt_dev_clean_512/fp16"
    use_graphs = True
    audio_batch_size = 1024
    audio_buffer_num_lines = 4096
    audio_fp16_input = True
    dali_batches_issue_ahead = 0
    dali_pipeline_depth = 2
    gpu_batch_size = 1792
    gpu_copy_streams = 1
    nobatch_sorting = True
    num_warmups = 20480
    server_target_qps = 43400
    scenario = Scenario.Server
    benchmark = Benchmark.RNNT
    power_limit = 250


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx8(BenchmarkConfiguration):
    system = System("A100-SXM-80GB", Architecture.Ampere, 8)
    gpu_inference_streams = 1
    input_dtype = "fp16"
    input_format = "linear"
    map_path = "data_maps/rnnt_dev_clean_512/val_map.txt"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/rnnt_dev_clean_512/fp16"
    use_graphs = True
    audio_batch_size = 1024
    audio_buffer_num_lines = 4096
    audio_fp16_input = True
    dali_batches_issue_ahead = 0
    dali_pipeline_depth = 1
    gpu_batch_size = 2048
    gpu_copy_streams = 1
    nobatch_sorting = True
    num_warmups = 20480
    server_num_issue_query_threads = 0
    server_target_qps = 104000
    start_from_device = True
    scenario = Scenario.Server
    benchmark = Benchmark.RNNT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_SXM_80GBx8_MaxQ(BenchmarkConfiguration):
    system = System("A100-SXM-80GB", Architecture.Ampere, 8)
    gpu_inference_streams = 1
    input_dtype = "fp16"
    input_format = "linear"
    map_path = "data_maps/rnnt_dev_clean_512/val_map.txt"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/rnnt_dev_clean_512/fp16"
    use_graphs = True
    audio_batch_size = 1024
    audio_buffer_num_lines = 4096
    audio_fp16_input = True
    dali_batches_issue_ahead = 0
    dali_pipeline_depth = 1
    gpu_batch_size = 2048
    gpu_copy_streams = 1
    nobatch_sorting = True
    num_warmups = 20480
    server_num_issue_query_threads = 0
    server_target_qps = 88000
    start_from_device = True
    scenario = Scenario.Server
    benchmark = Benchmark.RNNT
    power_limit = 275


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_40GBx1(BenchmarkConfiguration):
    system = System("A100-SXM4-40GB", Architecture.Ampere, 1)
    gpu_inference_streams = 1
    input_dtype = "fp16"
    input_format = "linear"
    map_path = "data_maps/rnnt_dev_clean_512/val_map.txt"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/rnnt_dev_clean_512/fp16"
    use_graphs = True
    audio_batch_size = 1024
    audio_buffer_num_lines = 4096
    audio_fp16_input = True
    dali_batches_issue_ahead = 0
    dali_pipeline_depth = 2
    gpu_batch_size = 1792
    gpu_copy_streams = 1
    nobatch_sorting = True
    num_warmups = 20480
    server_target_qps = 12750
    start_from_device = True
    scenario = Scenario.Server
    benchmark = Benchmark.RNNT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_40GBx8(BenchmarkConfiguration):
    system = System("A100-SXM4-40GB", Architecture.Ampere, 8)
    gpu_inference_streams = 1
    input_dtype = "fp16"
    input_format = "linear"
    map_path = "data_maps/rnnt_dev_clean_512/val_map.txt"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/rnnt_dev_clean_512/fp16"
    use_graphs = True
    audio_batch_size = 1024
    audio_buffer_num_lines = 4096
    audio_fp16_input = True
    dali_batches_issue_ahead = 0
    dali_pipeline_depth = 1
    gpu_batch_size = 2048
    gpu_copy_streams = 1
    nobatch_sorting = True
    num_warmups = 20480
    server_num_issue_query_threads = 0
    server_target_qps = 90000
    start_from_device = True
    scenario = Scenario.Server
    benchmark = Benchmark.RNNT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A10x1(BenchmarkConfiguration):
    system = System("A10", Architecture.Ampere, 1)
    gpu_inference_streams = 2
    input_dtype = "fp16"
    input_format = "linear"
    map_path = "data_maps/rnnt_dev_clean_512/val_map.txt"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/rnnt_dev_clean_512/fp16"
    use_graphs = True
    audio_batch_size = 128
    audio_buffer_num_lines = 4096
    dali_batches_issue_ahead = 0
    dali_pipeline_depth = 1
    gpu_batch_size = 512
    gpu_copy_streams = 2
    nobatch_sorting = True
    num_warmups = 2048
    server_target_qps = 3300
    scenario = Scenario.Server
    benchmark = Benchmark.RNNT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A10x8(BenchmarkConfiguration):
    system = System("A10", Architecture.Ampere, 8)
    gpu_inference_streams = 2
    input_dtype = "fp16"
    input_format = "linear"
    map_path = "data_maps/rnnt_dev_clean_512/val_map.txt"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/rnnt_dev_clean_512/fp16"
    use_graphs = True
    audio_batch_size = 128
    audio_buffer_num_lines = 4096
    dali_batches_issue_ahead = 0
    dali_pipeline_depth = 1
    gpu_batch_size = 512
    gpu_copy_streams = 2
    nobatch_sorting = True
    num_warmups = 2048
    server_target_qps = 22605.0
    scenario = Scenario.Server
    benchmark = Benchmark.RNNT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb(BenchmarkConfiguration):
    _mig_configuration = MIGConfiguration({0: {MIGSlice(1, 6): 1}})
    system = System("A30", Architecture.Ampere, 1, mig_conf=_mig_configuration)
    gpu_inference_streams = 1
    input_dtype = "fp16"
    input_format = "linear"
    map_path = "data_maps/rnnt_dev_clean_512/val_map.txt"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/rnnt_dev_clean_512/fp16"
    use_graphs = True
    audio_batch_size = 32
    audio_buffer_num_lines = 512
    dali_batches_issue_ahead = 1
    dali_pipeline_depth = 1
    gpu_batch_size = 256
    gpu_copy_streams = 1
    num_warmups = 32
    server_target_qps = 1100
    workspace_size = 1610612736
    scenario = Scenario.Server
    benchmark = Benchmark.RNNT


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_Hetero(A30_MIG_1x1g6gb):
    server_target_qps = 950


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x1(BenchmarkConfiguration):
    system = System("A30", Architecture.Ampere, 1)
    gpu_inference_streams = 1
    input_dtype = "fp16"
    input_format = "linear"
    map_path = "data_maps/rnnt_dev_clean_512/val_map.txt"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/rnnt_dev_clean_512/fp16"
    use_graphs = True
    audio_batch_size = 1024
    audio_buffer_num_lines = 4096
    audio_fp16_input = True
    dali_batches_issue_ahead = 0
    dali_pipeline_depth = 2
    gpu_batch_size = 1792
    gpu_copy_streams = 1
    nobatch_sorting = True
    num_warmups = 20480
    server_target_qps = 5200
    scenario = Scenario.Server
    benchmark = Benchmark.RNNT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x8(BenchmarkConfiguration):
    system = System("A30", Architecture.Ampere, 8)
    gpu_inference_streams = 1
    input_dtype = "fp16"
    input_format = "linear"
    map_path = "data_maps/rnnt_dev_clean_512/val_map.txt"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/rnnt_dev_clean_512/fp16"
    use_graphs = True
    audio_batch_size = 1024
    audio_buffer_num_lines = 4096
    audio_fp16_input = True
    dali_batches_issue_ahead = 0
    dali_pipeline_depth = 2
    gpu_batch_size = 1792
    gpu_copy_streams = 1
    nobatch_sorting = True
    num_warmups = 20480
    server_target_qps = 37000
    scenario = Scenario.Server
    benchmark = Benchmark.RNNT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A30x8_MaxQ(BenchmarkConfiguration):
    system = System("A30", Architecture.Ampere, 8)
    gpu_inference_streams = 1
    input_dtype = "fp16"
    input_format = "linear"
    map_path = "data_maps/rnnt_dev_clean_512/val_map.txt"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/rnnt_dev_clean_512/fp16"
    use_graphs = True
    audio_batch_size = 1024
    audio_buffer_num_lines = 4096
    audio_fp16_input = True
    dali_batches_issue_ahead = 0
    dali_pipeline_depth = 2
    gpu_batch_size = 1792
    gpu_copy_streams = 1
    nobatch_sorting = True
    num_warmups = 20480
    server_target_qps = 43500.0
    scenario = Scenario.Server
    benchmark = Benchmark.RNNT
    power_limit = 200


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x1(BenchmarkConfiguration):
    system = System("T4", Architecture.Turing, 1)
    gpu_inference_streams = 1
    input_dtype = "fp16"
    input_format = "linear"
    map_path = "data_maps/rnnt_dev_clean_512/val_map.txt"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/rnnt_dev_clean_512/fp16"
    use_graphs = True
    audio_batch_size = 64
    audio_buffer_num_lines = 512
    dali_batches_issue_ahead = 0
    dali_pipeline_depth = 1
    disable_encoder_plugin = True
    gpu_batch_size = 256
    gpu_copy_streams = 4
    max_seq_length = 102
    nobatch_sorting = True
    num_warmups = 2048
    server_target_qps = 1050
    scenario = Scenario.Server
    benchmark = Benchmark.RNNT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x20(BenchmarkConfiguration):
    system = System("T4", Architecture.Turing, 20)
    gpu_inference_streams = 1
    input_dtype = "fp16"
    input_format = "linear"
    map_path = "data_maps/rnnt_dev_clean_512/val_map.txt"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/rnnt_dev_clean_512/fp16"
    use_graphs = True
    audio_batch_size = 64
    audio_buffer_num_lines = 512
    dali_batches_issue_ahead = 0
    dali_pipeline_depth = 1
    disable_encoder_plugin = True
    gpu_batch_size = 256
    gpu_copy_streams = 4
    max_seq_length = 102
    nobatch_sorting = True
    num_warmups = 20480
    server_target_qps = 17000
    scenario = Scenario.Server
    benchmark = Benchmark.RNNT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x8(BenchmarkConfiguration):
    system = System("T4", Architecture.Turing, 8)
    gpu_inference_streams = 1
    input_dtype = "fp16"
    input_format = "linear"
    map_path = "data_maps/rnnt_dev_clean_512/val_map.txt"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/rnnt_dev_clean_512/fp16"
    use_graphs = True
    audio_batch_size = 64
    audio_buffer_num_lines = 640
    dali_batches_issue_ahead = 0
    dali_pipeline_depth = 1
    disable_encoder_plugin = True
    gpu_batch_size = 320
    gpu_copy_streams = 4
    max_seq_length = 102
    nobatch_sorting = True
    num_warmups = 20480
    server_target_qps = 8100
    scenario = Scenario.Server
    benchmark = Benchmark.RNNT
