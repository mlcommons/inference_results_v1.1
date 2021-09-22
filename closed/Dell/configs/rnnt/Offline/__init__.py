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
    gpu_inference_streams = 1
    input_dtype = "fp16"
    input_format = "linear"
    map_path = "data_maps/rnnt_dev_clean_512/val_map.txt"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/rnnt_dev_clean_512/fp16"
    use_graphs = True
    audio_batch_size = 512
    audio_buffer_num_lines = 4096
    dali_batches_issue_ahead = 4
    dali_pipeline_depth = 4
    gpu_batch_size = 2048
    gpu_copy_streams = 1
    offline_expected_qps = 12000*10
    num_warmups = 512
    scenario = Scenario.Offline
    benchmark = Benchmark.RNNT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class DSS8440_A30x8(BenchmarkConfiguration):
    system = System("DSS8440_A30", Architecture.Ampere, 8)
    gpu_inference_streams = 1
    input_dtype = "fp16"
    input_format = "linear"
    map_path = "data_maps/rnnt_dev_clean_512/val_map.txt"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/rnnt_dev_clean_512/fp16"
    use_graphs = True
    audio_batch_size = 512
    audio_buffer_num_lines = 4096
    dali_batches_issue_ahead = 4
    dali_pipeline_depth = 4
    gpu_batch_size = 2048
    gpu_copy_streams = 1
    offline_expected_qps = 52154
    num_warmups = 512
    scenario = Scenario.Offline
    benchmark = Benchmark.RNNT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750xa_A100_PCIE_80GBx4(BenchmarkConfiguration):
    system = System("R750xa_A100-PCIE-80GB", Architecture.Ampere, 4)
    gpu_inference_streams = 1
    input_dtype = "fp16"
    input_format = "linear"
    map_path = "data_maps/rnnt_dev_clean_512/val_map.txt"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/rnnt_dev_clean_512/fp16"
    use_graphs = True
    audio_batch_size = 512
    audio_buffer_num_lines = 4096
    dali_batches_issue_ahead = 4
    dali_pipeline_depth = 4
    gpu_batch_size = 2048
    gpu_copy_streams = 1
    offline_expected_qps = 55000
    num_warmups = 512
    scenario = Scenario.Offline
    benchmark = Benchmark.RNNT
    start_from_device = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R7525_A100_PCIE_40GBx3(BenchmarkConfiguration):
    system = System("R7525_A100-PCIE-40GB", Architecture.Ampere, 3)
    gpu_inference_streams = 2
    input_dtype = "fp16"
    input_format = "linear"
    map_path = "data_maps/rnnt_dev_clean_512/val_map.txt"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/rnnt_dev_clean_512/fp16"
    use_graphs = True
    audio_batch_size = 512
    audio_buffer_num_lines = 4096
    dali_batches_issue_ahead = 4
    dali_pipeline_depth = 4
    gpu_batch_size = 2111
    gpu_copy_streams = 8
    offline_expected_qps = 36728
    num_warmups = 512
    scenario = Scenario.Offline
    benchmark = Benchmark.RNNT
    start_from_device=True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R7525_A30x3(BenchmarkConfiguration):
    system = System("R7525_A30", Architecture.Ampere, 3)
    gpu_inference_streams = 1
    input_dtype = "fp16"
    input_format = "linear"
    map_path = "data_maps/rnnt_dev_clean_512/val_map.txt"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/rnnt_dev_clean_512/fp16"
    use_graphs = True
    audio_batch_size = 512
    audio_buffer_num_lines = 4096
    dali_batches_issue_ahead = 4
    dali_pipeline_depth = 4
    gpu_batch_size = 2048
    gpu_copy_streams = 1
    offline_expected_qps = 6959.999999999999*3
    num_warmups = 512
    scenario = Scenario.Offline
    benchmark = Benchmark.RNNT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE2420_A10x1(BenchmarkConfiguration):
    system = System("XE2420_A10", Architecture.Ampere, 1)
    gpu_inference_streams = 1
    input_dtype = "fp16"
    input_format = "linear"
    map_path = "data_maps/rnnt_dev_clean_512/val_map.txt"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/rnnt_dev_clean_512/fp16"
    use_graphs = True
    audio_batch_size = 512
    audio_buffer_num_lines = 4096
    dali_batches_issue_ahead = 2
    dali_pipeline_depth = 2
    gpu_batch_size = 2048
    gpu_copy_streams = 1
    offline_expected_qps = 4500
    num_warmups = 512
    scenario = Scenario.Offline
    benchmark = Benchmark.RNNT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class XE2420_A10x1_MaxQ(XE2420_A10x1):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE2420_A10x2(XE2420_A10x1):
    system = System("XE2420_A10", Architecture.Ampere, 2)
    gpu_inference_streams = 1
    input_dtype = "fp16"
    input_format = "linear"
    map_path = "data_maps/rnnt_dev_clean_512/val_map.txt"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/rnnt_dev_clean_512/fp16"
    use_graphs = True
    audio_batch_size = 512
    audio_buffer_num_lines = 4096
    dali_batches_issue_ahead = 2
    dali_pipeline_depth = 2
    gpu_batch_size = 2048
    gpu_copy_streams = 1
    offline_expected_qps = 10000
    num_warmups = 512
    scenario = Scenario.Offline
    benchmark = Benchmark.RNNT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class XE2420_A10x2_MaxQ(XE2420_A10x2):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8545_A100_SXM_80GBx4(BenchmarkConfiguration):
    system = System("XE8545_A100-SXM-80GB", Architecture.Ampere, 4)
    gpu_inference_streams = 1
    input_dtype = "fp16"
    input_format = "linear"
    map_path = "data_maps/rnnt_dev_clean_512/val_map.txt"
    precision = "fp16"
    tensor_path = "${PREPROCESSED_DATA_DIR}/rnnt_dev_clean_512/fp16"
    use_graphs = True
    audio_batch_size = 1024
    dali_batches_issue_ahead = 0
    dali_pipeline_depth = 1
    gpu_batch_size = 2048
    gpu_copy_streams = 1
    offline_expected_qps = 55000
    num_warmups = 40480
    audio_buffer_num_lines = 4096
    nobatch_sorting = True
    scenario = Scenario.Offline
    benchmark = Benchmark.RNNT

