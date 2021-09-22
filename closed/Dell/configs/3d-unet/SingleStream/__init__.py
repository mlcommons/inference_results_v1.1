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

from importlib import import_module
from code.common.constants import Benchmark, Scenario
from code.common.system_list import System, Architecture, MIGConfiguration, MIGSlice
from configs.configuration import *

ParentConfig = import_module("configs.3d-unet")
GPUBaseConfig = ParentConfig.GPUBaseConfig


class SingleStreamGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.SingleStream

    input_dtype = "fp16"
    input_format = "dhwc8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/brats/brats_npy/fp16_dhwc8"
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE2420_A10x1(SingleStreamGPUBaseConfig):
    system = System("XE2420_A10", Architecture.Ampere, 1)
    single_stream_expected_latency_ns = 68000000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class XE2420_A10x1_MaxQ(XE2420_A10x1):
    pass


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE2420_A10x1_HighAccuracy(XE2420_A10x1):
    pass


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class XE2420_A10x1_HighAccuracy_MaxQ(XE2420_A10x1_HighAccuracy):
    pass


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR12_A10x2(SingleStreamGPUBaseConfig):
    system = System("XR12_A10", Architecture.Ampere, 2)
    single_stream_expected_latency_ns = 68000000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XR12_A10x2_HighAccuracy(XR12_A10x2):
    pass

