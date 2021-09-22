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
from configs.bert import GPUBaseConfig, CPUBaseConfig


class OfflineGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.Offline

    gpu_copy_streams = 2
    gpu_inference_streams = 2
    enable_interleaved = False


class OfflineCPUBaseConfig(CPUBaseConfig):
    scenario = Scenario.Offline

    max_queue_delay_usec = 100


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx1(OfflineGPUBaseConfig):
    system = System("A100-PCIe-80GB", Architecture.Ampere, 1)
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 1024
    offline_expected_qps = 3400
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_80GBx1_HighAccuracy(A100_PCIe_80GBx1):
    precision = "fp16"
    offline_expected_qps = 1750


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx1_Triton(A100_PCIe_80GBx1):
    use_triton = True
    offline_expected_qps = 3000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_80GBx1_HighAccuracy_Triton(A100_PCIe_80GBx1_HighAccuracy):
    use_triton = True
    offline_expected_qps = 1550


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx8(A100_PCIe_80GBx1):
    system = System("A100-PCIe-80GB", Architecture.Ampere, 8)
    offline_expected_qps = 27200


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_80GBx8_HighAccuracy(A100_PCIe_80GBx8):
    precision = "fp16"
    offline_expected_qps = 12800


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx8_Triton(A100_PCIe_80GBx8):
    use_triton = True
    offline_expected_qps = 27000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_80GBx8_HighAccuracy_Triton(A100_PCIe_80GBx8_HighAccuracy):
    use_triton = True
    offline_expected_qps = 12800


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_PCIe_80GBx8_MaxQ(A100_PCIe_80GBx8):
    offline_expected_qps = 27200
    power_limit = 225


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class A100_PCIe_80GBx8_HighAccuracy_MaxQ(A100_PCIe_80GBx8_MaxQ):
    precision = "fp16"
    offline_expected_qps = 11000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_PCIe_80GBx8_Triton_MaxQ(A100_PCIe_80GBx8_MaxQ):
    use_triton = True
    offline_expected_qps = 27200


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class A100_PCIe_80GBx8_HighAccuracy_Triton_MaxQ(A100_PCIe_80GBx8_HighAccuracy_MaxQ):
    use_triton = True
    offline_expected_qps = 11168


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x1(OfflineGPUBaseConfig):
    system = System("A100-PCIe-80GB", Architecture.Ampere, 1, cpu_arch=CPUArch.aarch64)
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 1024
    offline_expected_qps = 3400
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x1_HighAccuracy(A100_PCIe_80GB_aarch64x1):
    precision = "fp16"
    offline_expected_qps = 1950


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x2(OfflineGPUBaseConfig):
    system = System("A100-PCIe-80GB", Architecture.Ampere, 2, cpu_arch=CPUArch.aarch64)
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 1024
    offline_expected_qps = 6500
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x2_HighAccuracy(A100_PCIe_80GB_aarch64x2):
    precision = "fp16"
    offline_expected_qps = 3900


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x4(OfflineGPUBaseConfig):
    system = System("A100-PCIe-80GB", Architecture.Ampere, 4, cpu_arch=CPUArch.aarch64)
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 1024
    offline_expected_qps = 13600
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x4_HighAccuracy(A100_PCIe_80GB_aarch64x4):
    precision = "fp16"
    offline_expected_qps = 8200


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_PCIe_80GB_aarch64x4_MaxQ(A100_PCIe_80GB_aarch64x4):
    offline_expected_qps = 9000
    power_limit = 225


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class A100_PCIe_80GB_aarch64x4_HighAccuracy_MaxQ(A100_PCIe_80GB_aarch64x4_MaxQ):
    precision = "fp16"
    offline_expected_qps = 4500


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_aarch64x1(OfflineGPUBaseConfig):
    system = System("A100-PCIe", Architecture.Ampere, 1, cpu_arch=CPUArch.aarch64)
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 1024
    offline_expected_qps = 3400
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_aarch64x1_HighAccuracy(A100_PCIe_aarch64x1):
    precision = "fp16"
    offline_expected_qps = 1950


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_aarch64x2(OfflineGPUBaseConfig):
    system = System("A100-PCIe", Architecture.Ampere, 2, cpu_arch=CPUArch.aarch64)
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 1024
    offline_expected_qps = 6500
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_aarch64x2_HighAccuracy(A100_PCIe_aarch64x2):
    precision = "fp16"
    offline_expected_qps = 3900


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_aarch64x4(OfflineGPUBaseConfig):
    system = System("A100-PCIe", Architecture.Ampere, 4, cpu_arch=CPUArch.aarch64)
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 1024
    offline_expected_qps = 13600
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_aarch64x4_HighAccuracy(A100_PCIe_aarch64x4):
    precision = "fp16"
    offline_expected_qps = 8200


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_PCIe_aarch64x4_MaxQ(A100_PCIe_aarch64x4):
    offline_expected_qps = 9000
    power_limit = 225


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class A100_PCIe_aarch64x4_HighAccuracy_MaxQ(A100_PCIe_aarch64x4_MaxQ):
    precision = "fp16"
    offline_expected_qps = 4500


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_MIG_1x1g5gb(OfflineGPUBaseConfig):
    _mig_configuration = MIGConfiguration({0: {MIGSlice(1, 5): 1}})
    system = System("A100-PCIe", Architecture.Ampere, 1, mig_conf=_mig_configuration)
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 64
    offline_expected_qps = 500
    workspace_size = 2147483648


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_MIG_1x1g5gb_HighAccuracy(A100_PCIe_MIG_1x1g5gb):
    precision = "fp16"
    offline_expected_qps = 225


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_MIG_1x1g5gb_Triton(A100_PCIe_MIG_1x1g5gb):
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_MIG_1x1g5gb_HighAccuracy_Triton(A100_PCIe_MIG_1x1g5gb_HighAccuracy):
    use_triton = True
    gpu_batch_size = 32
    offline_expected_qps = 225


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIex1(OfflineGPUBaseConfig):
    system = System("A100-PCIe", Architecture.Ampere, 1)
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 1024
    offline_expected_qps = 3400
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIex1_HighAccuracy(A100_PCIex1):
    precision = "fp16"
    offline_expected_qps = 1750


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIex1_Triton(A100_PCIex1):
    use_triton = True
    offline_expected_qps = 3000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIex1_HighAccuracy_Triton(A100_PCIex1_HighAccuracy):
    use_triton = True
    offline_expected_qps = 1750


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIex8(A100_PCIex1):
    system = System("A100-PCIe", Architecture.Ampere, 8)
    offline_expected_qps = 27200


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIex8_HighAccuracy(A100_PCIex8):
    precision = "fp16"
    offline_expected_qps = 11168


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIex8_Triton(A100_PCIex8):
    use_triton = True
    offline_expected_qps = 27000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIex8_HighAccuracy_Triton(A100_PCIex8_HighAccuracy):
    use_triton = True
    offline_expected_qps = 15000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_PCIex8_MaxQ(A100_PCIex8):
    offline_expected_qps = 27200
    power_limit = 225


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class A100_PCIex8_HighAccuracy_MaxQ(A100_PCIex8_MaxQ):
    precision = "fp16"
    offline_expected_qps = 11168


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_PCIex8_Triton_MaxQ(A100_PCIex8_MaxQ):
    use_triton = True
    offline_expected_qps = 27200


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class A100_PCIex8_HighAccuracy_Triton_MaxQ(A100_PCIex8_HighAccuracy_MaxQ):
    use_triton = True
    offline_expected_qps = 11168


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb(OfflineGPUBaseConfig):
    _mig_configuration = MIGConfiguration({0: {MIGSlice(1, 10): 1}})
    system = System("A100-SXM-80GB", Architecture.Ampere, 1, mig_conf=_mig_configuration)
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 64
    offline_expected_qps = 500
    workspace_size = 2147483648


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb_Hetero(A100_SXM_80GB_MIG_1x1g10gb):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb_HighAccuracy(A100_SXM_80GB_MIG_1x1g10gb):
    precision = "fp16"
    offline_expected_qps = 225


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb_Hetero_HighAccuracy(A100_SXM_80GB_MIG_1x1g10gb_HighAccuracy):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb_Triton(A100_SXM_80GB_MIG_1x1g10gb):
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb_HighAccuracy_Triton(A100_SXM_80GB_MIG_1x1g10gb_HighAccuracy):
    use_triton = True
    gpu_batch_size = 32
    offline_expected_qps = 225


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_56x1g10gb(OfflineGPUBaseConfig):
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
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 64
    offline_expected_qps = 28000
    workspace_size = 2147483648


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_56x1g10gb_HighAccuracy(A100_SXM_80GB_MIG_56x1g10gb):
    precision = "fp16"
    offline_expected_qps = 12600


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_56x1g10gb_Triton(A100_SXM_80GB_MIG_56x1g10gb):
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_56x1g10gb_HighAccuracy_Triton(A100_SXM_80GB_MIG_56x1g10gb_HighAccuracy):
    use_triton = True
    gpu_batch_size = 32
    offline_expected_qps = 12600


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx1(OfflineGPUBaseConfig):
    system = System("A100-SXM-80GB", Architecture.Ampere, 1)
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 1280
    gpu_inference_streams = 1
    offline_expected_qps = 3500


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GBx1_HighAccuracy(A100_SXM_80GBx1):
    precision = "fp16"
    gpu_batch_size = 512
    offline_expected_qps = 1750


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx1_Triton(A100_SXM_80GBx1):
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GBx1_HighAccuracy_Triton(A100_SXM_80GBx1_HighAccuracy):
    use_triton = True
    gpu_batch_size = 1280
    offline_expected_qps = 1750


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx4(OfflineGPUBaseConfig):
    _system_alias = "DGX Station A100 - Red October"
    _notes = "This should not inherit from A100_SXM_80GB (DGX-A100), and cannot use start_from_device"

    system = System("A100-SXM-80GB", Architecture.Ampere, 4)
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 1280
    gpu_inference_streams = 1
    offline_expected_qps = 12000
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GBx4_HighAccuracy(A100_SXM_80GBx4):
    precision = "fp16"
    offline_expected_qps = 5700


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx4_Triton(A100_SXM_80GBx4):
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GBx4_HighAccuracy_Triton(A100_SXM_80GBx4_HighAccuracy):
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_SXM_80GBx4_MaxQ(A100_SXM_80GBx4):
    offline_expected_qps = 10000
    power_limit = 250


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class A100_SXM_80GBx4_HighAccuracy_MaxQ(A100_SXM_80GBx4_MaxQ):
    precision = "fp16"
    offline_expected_qps = 5000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_SXM_80GBx4_Triton_MaxQ(A100_SXM_80GBx4_MaxQ):
    use_triton = True
    offline_expected_qps = 5000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class A100_SXM_80GBx4_HighAccuracy_Triton_MaxQ(A100_SXM_80GBx4_HighAccuracy_MaxQ):
    use_triton = True
    offline_expected_qps = 10000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx8(A100_SXM_80GBx1):
    system = System("A100-SXM-80GB", Architecture.Ampere, 8)
    offline_expected_qps = 30000
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GBx8_HighAccuracy(A100_SXM_80GBx8):
    precision = "fp16"
    gpu_batch_size = 512
    offline_expected_qps = 15000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx8_Triton(A100_SXM_80GBx8):
    use_triton = True
    offline_expected_qps = 29000
    workspace_size = 7516192768
    batch_triton_requests = False


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GBx8_HighAccuracy_Triton(A100_SXM_80GBx8_HighAccuracy):
    use_triton = True
    gpu_batch_size = 1280
    gpu_inference_streams = 1
    offline_expected_qps = 15000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_SXM_80GBx8_MaxQ(A100_SXM_80GBx8):
    power_limit = 300


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class A100_SXM_80GBx8_HighAccuracy_MaxQ(A100_SXM_80GBx8_MaxQ):
    precision = "fp16"
    gpu_batch_size = 512
    offline_expected_qps = 15000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_SXM_80GBx8_Triton_MaxQ(A100_SXM_80GBx8_MaxQ):
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class A100_SXM_80GBx8_HighAccuracy_Triton_MaxQ(A100_SXM_80GBx8_HighAccuracy_MaxQ):
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_40GB_MIG_1x1g5gb(OfflineGPUBaseConfig):
    _mig_configuration = MIGConfiguration({0: {MIGSlice(1, 5): 1}})
    system = System("A100-SXM4-40GB", Architecture.Ampere, 1, mig_conf=_mig_configuration)
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 64
    offline_expected_qps = 500
    workspace_size = 2147483648


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM4_40GB_MIG_1x1g5gb_HighAccuracy(A100_SXM4_40GB_MIG_1x1g5gb):
    precision = "fp16"
    gpu_batch_size = 64
    offline_expected_qps = 225


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_40GB_MIG_1x1g5gb_Triton(A100_SXM4_40GB_MIG_1x1g5gb):
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM4_40GB_MIG_1x1g5gb_HighAccuracy_Triton(A100_SXM4_40GB_MIG_1x1g5gb_HighAccuracy):
    use_triton = True
    gpu_batch_size = 32
    offline_expected_qps = 225


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_40GBx1(OfflineGPUBaseConfig):
    system = System("A100-SXM4-40GB", Architecture.Ampere, 1)
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 1024
    offline_expected_qps = 3400


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM4_40GBx1_HighAccuracy(A100_SXM4_40GBx1):
    precision = "fp16"
    offline_expected_qps = 1750


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_40GBx1_Triton(A100_SXM4_40GBx1):
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM4_40GBx1_HighAccuracy_Triton(A100_SXM4_40GBx1_HighAccuracy):
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_40GBx8(A100_SXM4_40GBx1):
    system = System("A100-SXM4-40GB", Architecture.Ampere, 8)
    offline_expected_qps = 30000
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM4_40GBx8_HighAccuracy(A100_SXM4_40GBx8):
    precision = "fp16"
    gpu_batch_size = 1024
    offline_expected_qps = 15000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_40GBx8_Triton(A100_SXM4_40GBx8):
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM4_40GBx8_HighAccuracy_Triton(A100_SXM4_40GBx8_HighAccuracy):
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A10x1(OfflineGPUBaseConfig):
    system = System("A10", Architecture.Ampere, 1)
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 256
    offline_expected_qps = 1100


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A10x1_HighAccuracy(A10x1):
    precision = "fp16"
    gpu_inference_streams = 1
    offline_expected_qps = 532


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A10x1_Triton(A10x1):
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A10x1_HighAccuracy_Triton(A10x1_HighAccuracy):
    use_triton = True
    gpu_inference_streams = 2
    gpu_batch_size = 256
    offline_expected_qps = 532


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A10x8(A10x1):
    system = System("A10", Architecture.Ampere, 8)
    offline_expected_qps = 8800


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A10x8_HighAccuracy(A10x8):
    precision = "fp16"
    gpu_inference_streams = 1
    offline_expected_qps = 4256


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A10x8_Triton(A10x8):
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A10x8_HighAccuracy_Triton(A10x8_HighAccuracy):
    use_triton = True
    gpu_inference_streams = 2


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb(OfflineGPUBaseConfig):
    _mig_configuration = MIGConfiguration({0: {MIGSlice(1, 6): 1}})
    system = System("A30", Architecture.Ampere, 1, mig_conf=_mig_configuration)
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 96
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    offline_expected_qps = 505
    workspace_size = 805306368


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_HighAccuracy(A30_MIG_1x1g6gb):
    precision = "fp16"
    offline_expected_qps = 246.3


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_Hetero(A30_MIG_1x1g6gb):
    offline_expected_qps = 457.658


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_Hetero_HighAccuracy(A30_MIG_1x1g6gb_Hetero):
    precision = "fp16"
    offline_expected_qps = 219.18


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_Triton(A30_MIG_1x1g6gb):
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_HighAccuracy_Triton(A30_MIG_1x1g6gb_HighAccuracy):
    use_triton = True
    gpu_batch_size = 64
    offline_expected_qps = 240


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_32x1g6gb(A30_MIG_1x1g6gb):
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
    offline_expected_qps = 16064


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30_MIG_32x1g6gb_HighAccuracy(A30_MIG_32x1g6gb):
    precision = "fp16"
    offline_expected_qps = 8000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_32x1g6gb_Triton(A30_MIG_32x1g6gb):
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30_MIG_32x1g6gb_HighAccuracy_Triton(A30_MIG_32x1g6gb_HighAccuracy):
    use_triton = True
    gpu_batch_size = 64
    offline_expected_qps = 7680


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x1(OfflineGPUBaseConfig):
    system = System("A30", Architecture.Ampere, 1)
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 1024
    offline_expected_qps = 1971.9999999999998
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30x1_HighAccuracy(A30x1):
    precision = "fp16"
    offline_expected_qps = 1014.9999999999999


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x1_Triton(A30x1):
    use_triton = True
    offline_expected_qps = 1739.9999999999998


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30x1_HighAccuracy_Triton(A30x1_HighAccuracy):
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x8(A30x1):
    system = System("A30", Architecture.Ampere, 8)
    offline_expected_qps = 13000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30x8_HighAccuracy(A30x8):
    precision = "fp16"
    offline_expected_qps = 8119.999999999999


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x8_Triton(A30x8):
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30x8_HighAccuracy_Triton(A30x8_HighAccuracy):
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class AGX_Xavier(OfflineGPUBaseConfig):
    system = System("AGX_Xavier", Architecture.Xavier, 1, cpu_arch=CPUArch.aarch64)
    enable_interleaved = True
    use_small_tile_gemm_plugin = False
    gpu_batch_size = 8
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    offline_expected_qps = 97


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class AGX_Xavier_HighAccuracy(AGX_Xavier):
    precision = "fp16"
    offline_expected_qps = 50


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class AGX_Xavier_Triton(AGX_Xavier):
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class AGX_Xavier_MaxQ(AGX_Xavier):
    offline_expected_qps = 61

    # power settings
    xavier_gpu_freq = 828750000
    xavier_dla_freq = 115200000
    xavier_cpu_freq = 1190400
    xavier_emc_freq = 1600000000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class AGX_Xavier_HighAccuracy_MaxQ(AGX_Xavier_MaxQ):
    precision = "fp16"
    offline_expected_qps = 31


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class Xavier_NX(OfflineGPUBaseConfig):
    system = System("Xavier_NX", Architecture.Xavier, 1, cpu_arch=CPUArch.aarch64)
    enable_interleaved = True
    use_small_tile_gemm_plugin = False
    gpu_batch_size = 8
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    offline_expected_qps = 62
    workspace_size = 1073741824


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class Xavier_NX_HighAccuracy(Xavier_NX):
    precision = "fp16"
    offline_expected_qps = 23


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class Xavier_NX_Triton(Xavier_NX):
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class Xavier_NX_MaxQ(Xavier_NX):
    offline_expected_qps = 50

    # power settings
    xavier_gpu_freq = 854250000
    xavier_dla_freq = 115200000
    xavier_cpu_freq = 1190400
    xavier_emc_freq = 1331200000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class Xavier_NX_HighAccuracy_MaxQ(Xavier_NX_MaxQ):
    precision = "fp16"
    offline_expected_qps = 23


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x1(OfflineGPUBaseConfig):
    system = System("T4", Architecture.Turing, 1)
    enable_interleaved = True
    use_small_tile_gemm_plugin = False
    gpu_batch_size = 256
    offline_expected_qps = 430


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class T4x1_HighAccuracy(T4x1):
    precision = "fp16"
    gpu_inference_streams = 1
    offline_expected_qps = 210


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x1_Triton(T4x1):
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class T4x1_HighAccuracy_Triton(T4x1_HighAccuracy):
    use_triton = True
    gpu_inference_streams = 2
    offline_expected_qps = 189


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x20(T4x1):
    system = System("T4", Architecture.Turing, 20)
    enable_interleaved = True
    use_small_tile_gemm_plugin = False
    gpu_batch_size = 256
    offline_expected_qps = 8800


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class T4x20_HighAccuracy(T4x20):
    precision = "fp16"
    gpu_inference_streams = 1
    offline_expected_qps = 4400


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x20_Triton(T4x20):
    use_triton = True
    offline_expected_qps = 7920


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class T4x20_HighAccuracy_Triton(T4x20_HighAccuracy):
    use_triton = True
    gpu_inference_streams = 2
    offline_expected_qps = 3960


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x8(T4x1):
    system = System("T4", Architecture.Turing, 8)
    enable_interleaved = True
    use_small_tile_gemm_plugin = False
    gpu_batch_size = 256
    offline_expected_qps = 3500


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class T4x8_HighAccuracy(T4x8):
    precision = "fp16"
    gpu_inference_streams = 1
    offline_expected_qps = 1680


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x8_Triton(T4x8):
    use_triton = True
    offline_expected_qps = 3150


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class T4x8_HighAccuracy_Triton(T4x8_HighAccuracy):
    use_triton = True
    gpu_inference_streams = 2
    offline_expected_qps = 1512


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class Triton_CPU_2S_6258Rx1(OfflineCPUBaseConfig):
    system = System("Triton_CPU_2S_6258R", Architecture.Intel_CPU_x86_64, 1)
    batch_size = 0
    offline_expected_qps = 36
    num_instances = 28
    ov_parameters = {
        'CPU_THREADS_NUM': '56',
        'CPU_THROUGHPUT_STREAMS': '14',
        'ENABLE_BATCH_PADDING': 'NO',
        'SKIP_OV_DYNAMIC_BATCHSIZE': 'YES'
    }


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class Triton_CPU_4S_8380Hx1(OfflineCPUBaseConfig):
    system = System("Triton_CPU_4S_8380H", Architecture.Intel_CPU_x86_64, 1)
    batch_size = 0
    offline_expected_qps = 96
    num_instances = 56
    ov_parameters = {
        'CPU_THREADS_NUM': '112',
        'CPU_THROUGHPUT_STREAMS': '28',
        'ENABLE_BATCH_PADDING': 'NO',
        'SKIP_OV_DYNAMIC_BATCHSIZE': 'YES'
    }
