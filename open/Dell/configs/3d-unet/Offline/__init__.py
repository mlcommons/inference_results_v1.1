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


class OfflineGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.Offline
    gpu_inference_streams = 1
    gpu_copy_streams = 2


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class DSS8440_A30x8(GPUBaseConfig):
    system = System("DSS8440_A30", Architecture.Ampere, 8)
    gpu_batch_size = 2
    scenario = Scenario.Offline
    gpu_inference_streams = 1
    gpu_copy_streams = 2
    offline_expected_qps = 300
    numa_config = "0-3:0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46&4-7:1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47"
    end_on_device = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class DSS8440_A30x8_HighAccuracy(DSS8440_A30x8):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class DSS8440_A30x8_Triton(DSS8440_A30x8):
    input_dtype = "fp16"
    input_format = "dhwc8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/brats/brats_npy/fp16_dhwc8"
    gpu_batch_size = 2
    offline_expected_qps = 300
    use_triton = True
    output_pinned_memory = False


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class R750xa_A100_PCIE_40GBx4_MaxQ(GPUBaseConfig):
    system = System("R750xa_A100-PCIE-40GB", Architecture.Ampere, 4)
    gpu_batch_size = 2
    offline_expected_qps = 53*4
    scenario = Scenario.Offline
    gpu_inference_streams = 1
    gpu_copy_streams = 2
    power_limit=175


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class R750xa_A100_PCIE_40GBx4_HighAccuracy_MaxQ(GPUBaseConfig):
    system = System("R750xa_A100-PCIE-40GB", Architecture.Ampere, 4)
    gpu_batch_size = 2
    offline_expected_qps = 53*4
    scenario = Scenario.Offline
    gpu_inference_streams = 1
    gpu_copy_streams = 2
    power_limit=175


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750xa_A100_PCIE_80GBx4(GPUBaseConfig):
    scenario = Scenario.Offline
    gpu_inference_streams = 1
    gpu_copy_streams = 2
    system = System("R750xa_A100-PCIE-80GB", Architecture.Ampere, 4)
    gpu_batch_size = 2
    offline_expected_qps = 300
    end_on_device = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R750xa_A100_PCIE_80GBx4_HighAccuracy(R750xa_A100_PCIE_80GBx4):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750xa_A100_PCIE_80GBx4_Triton(R750xa_A100_PCIE_80GBx4):
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R750xa_A100_PCIE_80GBx4_HighAccuracy_Triton(R750xa_A100_PCIE_80GBx4_HighAccuracy):
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750xa_A100_PCIE_80GB_MIG_28x1g10gb_Triton(R750xa_A100_PCIE_80GBx4_HighAccuracy_Triton):
    _mig_configuration = MIGConfiguration({
        0: {MIGSlice(1, 10): 7},
        1: {MIGSlice(1, 10): 7},
        2: {MIGSlice(1, 10): 7},
        3: {MIGSlice(1, 10): 7},
    })
    system = System("R750xa_A100-PCIE-80GB", Architecture.Ampere, 4, mig_conf=_mig_configuration)
    gpu_copy_streams = 1
    workspace_size = 1073741824
    offline_expected_qps = 10


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R750xa_A100_PCIE_80GB_MIG_28x1g10gb_HighAccuracy_Triton(R750xa_A100_PCIE_80GB_MIG_28x1g10gb_Triton):
    pass


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R7525_A100_PCIE_40GBx3(GPUBaseConfig):
    system = System("R7525_A100-PCIE-40GB", Architecture.Ampere, 3)
    gpu_batch_size = 2
    offline_expected_qps = 53*3
    scenario = Scenario.Offline
    gpu_inference_streams = 1
    gpu_copy_streams = 2


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R7525_A100_PCIE_40GBx3(GPUBaseConfig):
    system = System("R7525_A100-PCIE-40GB", Architecture.Ampere, 3)
    gpu_batch_size = 2
    offline_expected_qps = 53*3
    scenario = Scenario.Offline
    gpu_inference_streams = 1
    gpu_copy_streams = 2


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R7525_A30x3(OfflineGPUBaseConfig):
    system = System("R7525_A30", Architecture.Ampere, 3)
    gpu_batch_size = 2
    offline_expected_qps = 30.74*3


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R7525_A30x3_HighAccuracy(R7525_A30x3):
    pass


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE2420_A10x1(OfflineGPUBaseConfig):
    system = System("XE2420_A10", Architecture.Ampere, 1)
    offline_expected_qps = 25
    gpu_batch_size = 2


# inherits XE2420_A10x1-99 values
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
class XE2420_A10x2(XE2420_A10x1):
    system = System("XE2420_A10", Architecture.Ampere, 2)
    offline_expected_qps = 50


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class XE2420_A10x2_MaxQ(XE2420_A10x2):
    pass


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE2420_A10x2_HighAccuracy(XE2420_A10x2):
    pass


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class XE2420_A10x2_HighAccuracy_MaxQ(XE2420_A10x2_HighAccuracy):
    pass


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8545_A100_SXM_80GBx4(OfflineGPUBaseConfig):
    system = System("XE8545_A100-SXM-80GB", Architecture.Ampere, 4)
    gpu_batch_size = 2
    offline_expected_qps = 260
    end_on_device = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE8545_A100_SXM_80GBx4_HighAccuracy(XE8545_A100_SXM_80GBx4):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8545_A100_SXM_80GBx4_Triton(OfflineGPUBaseConfig):
    end_on_device = True
    system = System("XE8545_A100-SXM-80GB", Architecture.Ampere, 4)
    gpu_batch_size = 2
    offline_expected_qps = 183 
    use_graphs = True
    instance_group_count = 4
    use_triton = True
    output_pinned_memory = False


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE8545_A100_SXM_80GBx4_HighAccuracy_Triton(XE8545_A100_SXM_80GBx4_Triton):
    pass


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR12_A10x2(OfflineGPUBaseConfig):
    system = System("XR12_A10", Architecture.Ampere, 2)
    offline_expected_qps = 50
    gpu_batch_size = 2


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XR12_A10x2_HighAccuracy(XR12_A10x2):
    pass

