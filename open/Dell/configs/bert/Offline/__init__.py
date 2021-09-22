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
from configs.bert import GPUBaseConfig

class OfflineGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.Offline

    gpu_copy_streams = 2
    gpu_inference_streams = 2
    enable_interleaved = False


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class DSS8440_A100_PCIE_80GBx10(GPUBaseConfig):
    system = System("DSS8440_A100-PCIE-80GB", Architecture.Ampere, 10)
    scenario = Scenario.Offline
    gpu_copy_streams = 2
    gpu_inference_streams = 2
    enable_interleaved = False
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 1024
    offline_expected_qps = 3400*10
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class DSS8440_A100_PCIE_80GBx10_HighAccuracy(DSS8440_A100_PCIE_80GBx10):
    precision = "fp16"
    offline_expected_qps = 1750*10


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class DSS8440_A30x8(GPUBaseConfig):
    system = System("DSS8440_A30", Architecture.Ampere, 8)
    offline_expected_qps = 14000
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 1024
    workspace_size = 7516192768
    scenario = Scenario.Offline
    gpu_copy_streams = 2
    gpu_inference_streams = 2
    enable_interleaved = False


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class DSS8440_A30x8_HighAccuracy(DSS8440_A30x8):
    precision = "fp16"
    offline_expected_qps = 8120


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class DSS8440_A30x8_Triton(DSS8440_A30x8):
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class DSS8440_A30x8_HighAccuracy_Triton(DSS8440_A30x8_HighAccuracy):
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class R750xa_A100_PCIE_40GBx4_MaxQ(GPUBaseConfig):
    system = System("R750xa_A100-PCIE-40GB", Architecture.Ampere, 4)
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 1024
    offline_expected_qps = 3400*4
    workspace_size = 7516192768
    scenario = Scenario.Offline
    gpu_copy_streams = 2
    gpu_inference_streams = 2
    enable_interleaved = False
    start_from_device = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class R750xa_A100_PCIE_40GBx4_HighAccuracy_MaxQ(R750xa_A100_PCIE_40GBx4_MaxQ):
    precision = "fp16"
    offline_expected_qps = 1750*4


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750xa_A100_PCIE_80GBx4(GPUBaseConfig):
    scenario = Scenario.Offline
    system = System("R750xa_A100-PCIE-80GB", Architecture.Ampere, 4)
    use_small_tile_gemm_plugin = True
    offline_expected_qps = 15000
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 1024
    workspace_size = 7516192768
    gpu_copy_streams = 2
    gpu_inference_streams = 2
    enable_interleaved = False


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R750xa_A100_PCIE_80GBx4_HighAccuracy(R750xa_A100_PCIE_80GBx4):
    precision = "fp16"
    offline_expected_qps = 7000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750xa_A100_PCIE_80GBx4_Triton(R750xa_A100_PCIE_80GBx4):
    use_triton = True
    offline_expected_qps = 13000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R750xa_A100_PCIE_80GBx4_HighAccuracy_Triton(R750xa_A100_PCIE_80GBx4_HighAccuracy):
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750xa_A100_PCIE_80GB_MIG_28x1g10gb_Triton(R750xa_A100_PCIE_80GBx4_Triton):
    _mig_configuration = MIGConfiguration({
        0: {MIGSlice(1, 10): 7},
        1: {MIGSlice(1, 10): 7},
        2: {MIGSlice(1, 10): 7},
        3: {MIGSlice(1, 10): 7},
    })
    system = System("R750xa_A100-PCIE-80GB", Architecture.Ampere, 4, mig_conf=_mig_configuration)
    gpu_batch_size = 64
    offline_expected_qps = 13000
    workspace_size = 2147483648


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R750xa_A100_PCIE_80GB_MIG_28x1g10gb_HighAccuracy_Triton(R750xa_A100_PCIE_80GBx4_HighAccuracy_Triton):
    _mig_configuration = MIGConfiguration({
        0: {MIGSlice(1, 10): 7},
        1: {MIGSlice(1, 10): 7},
        2: {MIGSlice(1, 10): 7},
        3: {MIGSlice(1, 10): 7},
    })
    system = System("R750xa_A100-PCIE-80GB", Architecture.Ampere, 4, mig_conf=_mig_configuration)
    gpu_batch_size = 32
    offline_expected_qps = 6600
    workspace_size = 2147483648


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R7525_A100_PCIE_40GBx3(GPUBaseConfig):
    system = System("R7525_A100-PCIE-40GB", Architecture.Ampere, 3)
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 1024
    offline_expected_qps = 3400*3
    workspace_size = 7516192768
    scenario = Scenario.Offline
    gpu_copy_streams = 2
    gpu_inference_streams = 2
    enable_interleaved = False
    start_from_device=True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R7525_A100_PCIE_40GBx3_HighAccuracy(R7525_A100_PCIE_40GBx3):
    precision = "fp16"
    offline_expected_qps = 1750*3


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R7525_A30x3(GPUBaseConfig):
    system = System("R7525_A30", Architecture.Ampere, 3)
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 1024
    offline_expected_qps = 1971.9999999999998*3
    workspace_size = 7516192768
    scenario = Scenario.Offline
    benchmark = Benchmark.BERT


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R7525_A30x3_HighAccuracy(R7525_A30x3):
    precision = "fp16"
    offline_expected_qps = 1014.9999999999999*3


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE2420_A10x1(OfflineGPUBaseConfig):
    system = System("XE2420_A10", Architecture.Ampere, 1)
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 256
    offline_expected_qps = 1200


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class XE2420_A10x1_MaxQ(XE2420_A10x1):
    pass


# inherits XE2420_A10x1-99 values
@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE2420_A10x2(XE2420_A10x1):
    system = System("XE2420_A10", Architecture.Ampere, 2)
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 256
    offline_expected_qps = 2950


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class XE2420_A10x2_MaxQ(XE2420_A10x2):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE2420_A10x2_HighAccuracy(XE2420_A10x2):
    precision = "fp16"
    gpu_inference_streams = 1
    offline_expected_qps = 2000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class XE2420_A10x2_HighAccuracy_MaxQ(XE2420_A10x2_HighAccuracy):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8545_A100_SXM_80GBx4(OfflineGPUBaseConfig):
    system = System("XE8545_A100-SXM-80GB", Architecture.Ampere, 4)
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 1280
    gpu_inference_streams = 1
    offline_expected_qps = 15000
    start_from_device = True
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE8545_A100_SXM_80GBx4_HighAccuracy(XE8545_A100_SXM_80GBx4):
    precision = "fp16"
    gpu_batch_size = 512
    offline_expected_qps = 7500


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8545_A100_SXM_80GBx4_Triton(XE8545_A100_SXM_80GBx4):
    use_triton = True
    offline_expected_qps = 15202
    batch_triton_requests = False
    start_from_device=False


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE8545_A100_SXM_80GBx4_HighAccuracy_Triton(XE8545_A100_SXM_80GBx4_HighAccuracy):
    use_triton = True
    gpu_batch_size = 1280
    gpu_inference_streams = 1
    offline_expected_qps = 7572
    start_from_device=False


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR12_A10x2(OfflineGPUBaseConfig):
    system = System("XR12_A10", Architecture.Ampere, 2)
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 256
    offline_expected_qps = 2950


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XR12_A10x2_HighAccuracy(XR12_A10x2):
    precision = "fp16"
    gpu_inference_streams = 1
    offline_expected_qps = 2000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_40GBx3(OfflineGPUBaseConfig):
    system = System("R7525_vA100-PCIE-40GB", Architecture.Ampere, 3)
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 1024
    offline_expected_qps = 10200
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class vA100_PCIe_40GBx3_HighAccuracy(A100_PCIe_40GBx3):
    precision = "fp16"
    offline_expected_qps = 5250
