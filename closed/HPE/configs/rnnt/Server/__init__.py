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
    server_target_qps = 52000
    scenario = Scenario.Server
    benchmark = Benchmark.RNNT
