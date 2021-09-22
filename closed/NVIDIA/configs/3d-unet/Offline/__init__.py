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
CPUBaseConfig = ParentConfig.CPUBaseConfig


class OfflineGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.Offline
    gpu_inference_streams = 1
    gpu_copy_streams = 2


class OfflineCPUBaseConfig(CPUBaseConfig):
    scenario = Scenario.Offline


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx1(OfflineGPUBaseConfig):
    system = System("A100-PCIe-80GB", Architecture.Ampere, 1)
    gpu_batch_size = 2
    offline_expected_qps = 53


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_80GBx1_HighAccuracy(A100_PCIe_80GBx1):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx1_Triton(A100_PCIe_80GBx1):
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_80GBx1_HighAccuracy_Triton(A100_PCIe_80GBx1_Triton):
    pass


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx8(A100_PCIe_80GBx1):
    system = System("A100-PCIe-80GB", Architecture.Ampere, 8)
    gpu_batch_size = 2
    offline_expected_qps = 412
    numa_config = "3:0-15&2:16-31&1:32-47&0:48-63&7:64-79&6:80-95&5:96-111&4:112-127"


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_80GBx8_HighAccuracy(A100_PCIe_80GBx8):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx8_Triton(A100_PCIe_80GBx8):
    input_dtype = "fp16"
    input_format = "dhwc8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/brats/brats_npy/fp16_dhwc8"
    use_triton = True
    output_pinned_memory = False


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_80GBx8_HighAccuracy_Triton(A100_PCIe_80GBx8_Triton):
    pass


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_PCIe_80GBx8_MaxQ(A100_PCIe_80GBx8):
    gpu_batch_size = 2
    offline_expected_qps = 370
    power_limit = 175
    numa_config = "3:0-7&2:8-15&1:16-23&0:24-31&7:32-39&6:40-47&5:48-55&4:56-63"


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class A100_PCIe_80GBx8_HighAccuracy_MaxQ(A100_PCIe_80GBx8_MaxQ):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_PCIe_80GBx8_Triton_MaxQ(A100_PCIe_80GBx8_MaxQ):
    use_triton = True
    offline_expected_qps = 412


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class A100_PCIe_80GBx8_HighAccuracy_Triton_MaxQ(A100_PCIe_80GBx8_Triton_MaxQ):
    pass


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x1(OfflineGPUBaseConfig):
    system = System("A100-PCIe-80GB", Architecture.Ampere, 1, cpu_arch=CPUArch.aarch64)
    gpu_batch_size = 2
    offline_expected_qps = 53


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x1_HighAccuracy(A100_PCIe_80GB_aarch64x1):
    pass


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x2(OfflineGPUBaseConfig):
    system = System("A100-PCIe-80GB", Architecture.Ampere, 2, cpu_arch=CPUArch.aarch64)
    gpu_batch_size = 2
    offline_expected_qps = 106


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x2_HighAccuracy(A100_PCIe_80GB_aarch64x2):
    pass


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x4(OfflineGPUBaseConfig):
    system = System("A100-PCIe-80GB", Architecture.Ampere, 4, cpu_arch=CPUArch.aarch64)
    gpu_batch_size = 2
    offline_expected_qps = 210


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x4_HighAccuracy(A100_PCIe_80GB_aarch64x4):
    pass


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_PCIe_80GB_aarch64x4_MaxQ(OfflineGPUBaseConfig):
    system = System("A100-PCIe-80GB", Architecture.Ampere, 4, cpu_arch=CPUArch.aarch64)
    gpu_batch_size = 2
    power_limit = 175
    offline_expected_qps = 185


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class A100_PCIe_80GB_aarch64x4_HighAccuracy_MaxQ(A100_PCIe_80GB_aarch64x4_MaxQ):
    pass


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_aarch64x1(OfflineGPUBaseConfig):
    system = System("A100-PCIe", Architecture.Ampere, 1, cpu_arch=CPUArch.aarch64)
    gpu_batch_size = 2
    offline_expected_qps = 53


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_aarch64x1_HighAccuracy(A100_PCIe_aarch64x1):
    pass


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_aarch64x2(OfflineGPUBaseConfig):
    system = System("A100-PCIe", Architecture.Ampere, 2, cpu_arch=CPUArch.aarch64)
    gpu_batch_size = 2
    offline_expected_qps = 106


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_aarch64x2_HighAccuracy(A100_PCIe_aarch64x2):
    pass


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_aarch64x4(OfflineGPUBaseConfig):
    system = System("A100-PCIe", Architecture.Ampere, 4, cpu_arch=CPUArch.aarch64)
    gpu_batch_size = 2
    offline_expected_qps = 210


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_aarch64x4_HighAccuracy(A100_PCIe_aarch64x4):
    pass


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_PCIe_aarch64x4_MaxQ(OfflineGPUBaseConfig):
    system = System("A100-PCIe", Architecture.Ampere, 4, cpu_arch=CPUArch.aarch64)
    gpu_batch_size = 2
    power_limit = 175
    offline_expected_qps = 185


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class A100_PCIe_aarch64x4_HighAccuracy_MaxQ(A100_PCIe_aarch64x4_MaxQ):
    pass


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_MIG_1x1g5gb(OfflineGPUBaseConfig):
    _mig_configuration = MIGConfiguration({0: {MIGSlice(1, 5): 1}})
    system = System("A100-PCIe", Architecture.Ampere, 1, mig_conf=_mig_configuration)
    input_dtype = "fp16"
    input_format = "linear"
    tensor_path = "${PREPROCESSED_DATA_DIR}/brats/brats_npy/fp16_linear"
    workspace_size = 1073741824
    gpu_batch_size = 1
    gpu_copy_streams = 1
    offline_expected_qps = 7


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_MIG_1x1g5gb_HighAccuracy(A100_PCIe_MIG_1x1g5gb):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_MIG_1x1g5gb_Triton(A100_PCIe_MIG_1x1g5gb):
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_MIG_1x1g5gb_HighAccuracy_Triton(A100_PCIe_MIG_1x1g5gb_Triton):
    pass


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIex1(OfflineGPUBaseConfig):
    system = System("A100-PCIe", Architecture.Ampere, 1)
    gpu_batch_size = 2
    offline_expected_qps = 53


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIex1_HighAccuracy(A100_PCIex1):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIex1_Triton(A100_PCIex1):
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIex1_HighAccuracy_Triton(A100_PCIex1_Triton):
    pass


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIex8(A100_PCIex1):
    system = System("A100-PCIe", Architecture.Ampere, 8)
    gpu_batch_size = 2
    offline_expected_qps = 412
    numa_config = "3:0-15&2:16-31&1:32-47&0:48-63&7:64-79&6:80-95&5:96-111&4:112-127"


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIex8_HighAccuracy(A100_PCIex8):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIex8_Triton(A100_PCIex8):
    input_dtype = "fp16"
    input_format = "dhwc8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/brats/brats_npy/fp16_dhwc8"
    use_triton = True
    output_pinned_memory = False


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIex8_HighAccuracy_Triton(A100_PCIex8_Triton):
    pass


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_PCIex8_MaxQ(A100_PCIex8):
    gpu_batch_size = 2
    offline_expected_qps = 370
    power_limit = 175
    numa_config = "3:0-7&2:8-15&1:16-23&0:24-31&7:32-39&6:40-47&5:48-55&4:56-63"


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class A100_PCIex8_HighAccuracy_MaxQ(A100_PCIex8_MaxQ):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_PCIex8_Triton_MaxQ(A100_PCIex8_MaxQ):
    gpu_batch_size = 2
    offline_expected_qps = 412
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class A100_PCIex8_HighAccuracy_Triton_MaxQ(A100_PCIex8_Triton_MaxQ):
    pass


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb(OfflineGPUBaseConfig):
    _mig_configuration = MIGConfiguration({0: {MIGSlice(1, 10): 1}})
    system = System("A100-SXM-80GB", Architecture.Ampere, 1, mig_conf=_mig_configuration)
    input_dtype = "fp16"
    input_format = "linear"
    tensor_path = "${PREPROCESSED_DATA_DIR}/brats/brats_npy/fp16_linear"
    workspace_size = 1073741824
    gpu_batch_size = 1
    gpu_copy_streams = 1
    offline_expected_qps = 7
    start_from_device = True


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb_Hetero(A100_SXM_80GB_MIG_1x1g10gb):
    pass


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb_HighAccuracy(A100_SXM_80GB_MIG_1x1g10gb):
    pass


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb_Hetero_HighAccuracy(A100_SXM_80GB_MIG_1x1g10gb_HighAccuracy):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb_Triton(A100_SXM_80GB_MIG_1x1g10gb):
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb_HighAccuracy_Triton(A100_SXM_80GB_MIG_1x1g10gb_Triton):
    pass


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_56x1g10gb(A100_SXM_80GB_MIG_1x1g10gb):
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
    gpu_batch_size = 1
    gpu_copy_streams = 1
    offline_expected_qps = 392


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_56x1g10gb_HighAccuracy(A100_SXM_80GB_MIG_56x1g10gb):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_56x1g10gb_Triton(A100_SXM_80GB_MIG_56x1g10gb):
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_56x1g10gb_HighAccuracy_Triton(A100_SXM_80GB_MIG_56x1g10gb_Triton):
    pass


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx1(OfflineGPUBaseConfig):
    system = System("A100-SXM-80GB", Architecture.Ampere, 1)
    gpu_batch_size = 2
    offline_expected_qps = 60
    start_from_device = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GBx1_HighAccuracy(A100_SXM_80GBx1):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx1_Triton(A100_SXM_80GBx1):
    instance_group_count = 1
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GBx1_HighAccuracy_Triton(A100_SXM_80GBx1_Triton):
    pass


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx4(OfflineGPUBaseConfig):
    _system_alias = "DGX Station A100 - Red October"
    _notes = "This should not inherit from A100_SXM_80GB (DGX-A100), and cannot use start_from_device"

    system = System("A100-SXM-80GB", Architecture.Ampere, 4)
    gpu_batch_size = 2
    offline_expected_qps = 220
    numa_config = "3:0-15,64-79&2:16-31,80-95&1:32-47,96-111&0:48-63,112-127"


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GBx4_HighAccuracy(A100_SXM_80GBx4):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx4_Triton(A100_SXM_80GBx4):
    instance_group_count = 1
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GBx4_HighAccuracy_Triton(A100_SXM_80GBx4_Triton):
    pass


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_SXM_80GBx4_MaxQ(A100_SXM_80GBx4):
    gpu_batch_size = 2
    offline_expected_qps = 220
    power_limit = 225
    numa_config = "3:0-7,32-39&2:8-15,40-47&1:16-23,48-55&0:24-31,56-63"


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class A100_SXM_80GBx4_HighAccuracy_MaxQ(A100_SXM_80GBx4_MaxQ):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_SXM_80GBx4_Triton_MaxQ(A100_SXM_80GBx4_MaxQ):
    numa_config = ""  # TODO: Artifact from old configs. Should Red October Triton use numa_config?
    instance_group_count = 1
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class A100_SXM_80GBx4_HighAccuracy_Triton_MaxQ(A100_SXM_80GBx4_Triton_MaxQ):
    pass


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx8(A100_SXM_80GBx1):
    system = System("A100-SXM-80GB", Architecture.Ampere, 8)
    gpu_batch_size = 2
    offline_expected_qps = 480


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GBx8_HighAccuracy(A100_SXM_80GBx8):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx8_Triton(A100_SXM_80GBx8):
    use_graphs = True
    instance_group_count = 4
    use_triton = True
    output_pinned_memory = False


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM_80GBx8_HighAccuracy_Triton(A100_SXM_80GBx8_Triton):
    pass


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_SXM_80GBx8_MaxQ(A100_SXM_80GBx8):
    gpu_batch_size = 2
    offline_expected_qps = 480
    power_limit = 225


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class A100_SXM_80GBx8_HighAccuracy_MaxQ(A100_SXM_80GBx8_MaxQ):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_SXM_80GBx8_Triton_MaxQ(A100_SXM_80GBx8_MaxQ):
    instance_group_count = 2
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class A100_SXM_80GBx8_HighAccuracy_Triton_MaxQ(A100_SXM_80GBx8_Triton_MaxQ):
    pass


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_40GB_MIG_1x1g5gb(OfflineGPUBaseConfig):
    _mig_configuration = MIGConfiguration({0: {MIGSlice(1, 5): 1}})
    system = System("A100-SXM4-40GB", Architecture.Ampere, 1, mig_conf=_mig_configuration)
    input_dtype = "fp16"
    input_format = "linear"
    tensor_path = "${PREPROCESSED_DATA_DIR}/brats/brats_npy/fp16_linear"
    workspace_size = 1073741824
    gpu_batch_size = 1
    gpu_copy_streams = 1
    offline_expected_qps = 7
    start_from_device = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM4_40GB_MIG_1x1g5gb_HighAccuracy(A100_SXM4_40GB_MIG_1x1g5gb):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_40GB_MIG_1x1g5gb_Triton(A100_SXM4_40GB_MIG_1x1g5gb):
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM4_40GB_MIG_1x1g5gb_HighAccuracy_Triton(A100_SXM4_40GB_MIG_1x1g5gb_Triton):
    pass


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_40GBx1(OfflineGPUBaseConfig):
    system = System("A100-SXM4-40GB", Architecture.Ampere, 1)
    gpu_batch_size = 2
    offline_expected_qps = 60
    start_from_device = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM4_40GBx1_HighAccuracy(A100_SXM4_40GBx1):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_40GBx1_Triton(A100_SXM4_40GBx1):
    instance_group_count = 1
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM4_40GBx1_HighAccuracy_Triton(A100_SXM4_40GBx1_Triton):
    pass


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_40GBx8(A100_SXM4_40GBx1):
    system = System("A100-SXM4-40GB", Architecture.Ampere, 8)
    offline_expected_qps = 480


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM4_40GBx8_HighAccuracy(A100_SXM4_40GBx8):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_40GBx8_Triton(A100_SXM4_40GBx8):
    instance_group_count = 4
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM4_40GBx8_HighAccuracy_Triton(A100_SXM4_40GBx8_Triton):
    pass


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_SXM4_40GBx8_MaxQ(A100_SXM4_40GBx8):
    power_limit = 225


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class A100_SXM4_40GBx8_HighAccuracy_MaxQ(A100_SXM4_40GBx8_MaxQ):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_SXM4_40GBx8_Triton_MaxQ(A100_SXM4_40GBx8_MaxQ):
    instance_group_count = 2
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class A100_SXM4_40GBx8_HighAccuracy_Triton_MaxQ(A100_SXM4_40GBx8_Triton_MaxQ):
    pass


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A10x1(OfflineGPUBaseConfig):
    system = System("A10", Architecture.Ampere, 1)
    gpu_batch_size = 2
    offline_expected_qps = 22


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A10x1_HighAccuracy(A10x1):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A10x1_Triton(A10x1):
    gpu_batch_size = 2
    offline_expected_qps = 20
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A10x1_HighAccuracy_Triton(A10x1_Triton):
    offline_expected_qps = 22


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A10x8(A10x1):
    system = System("A10", Architecture.Ampere, 8)
    offline_expected_qps = 170


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A10x8_HighAccuracy(A10x8):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A10x8_Triton(A10x8):
    gpu_batch_size = 2
    offline_expected_qps = 160.0
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A10x8_HighAccuracy_Triton(A10x8_Triton):
    offline_expected_qps = 170


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb(OfflineGPUBaseConfig):
    _mig_configuration = MIGConfiguration({0: {MIGSlice(1, 6): 1}})
    system = System("A30", Architecture.Ampere, 1, mig_conf=_mig_configuration)
    input_dtype = "fp16"
    input_format = "linear"
    tensor_path = "${PREPROCESSED_DATA_DIR}/brats/brats_npy/fp16_linear"
    workspace_size = 805306368
    gpu_batch_size = 1
    gpu_copy_streams = 1
    offline_expected_qps = 7.55


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_HighAccuracy(A30_MIG_1x1g6gb):
    pass


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_Hetero(A30_MIG_1x1g6gb):
    offline_expected_qps = 6.847


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_Hetero_HighAccuracy(A30_MIG_1x1g6gb_Hetero):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_Triton(A30_MIG_1x1g6gb):
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_HighAccuracy_Triton(A30_MIG_1x1g6gb_Triton):
    pass


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_32x1g6gb(OfflineGPUBaseConfig):
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
    input_dtype = "fp16"
    input_format = "linear"
    tensor_path = "${PREPROCESSED_DATA_DIR}/brats/brats_npy/fp16_linear"
    workspace_size = 805306368
    gpu_batch_size = 1
    gpu_copy_streams = 1
    offline_expected_qps = 224


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30_MIG_32x1g6gb_HighAccuracy(A30_MIG_32x1g6gb):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_32x1g6gb_Triton(A30_MIG_32x1g6gb):
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30_MIG_32x1g6gb_HighAccuracy_Triton(A30_MIG_32x1g6gb_Triton):
    pass


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x1(OfflineGPUBaseConfig):
    system = System("A30", Architecture.Ampere, 1)
    gpu_batch_size = 2
    offline_expected_qps = 30.74


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30x1_HighAccuracy(A30x1):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x1_Triton(A30x1):
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30x1_HighAccuracy_Triton(A30x1_Triton):
    pass


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x8(A30x1):
    system = System("A30", Architecture.Ampere, 8)
    offline_expected_qps = 230
    numa_config = "3:0-15&2:16-31&1:32-47&0:48-63&7:64-79&6:80-95&5:96-111&4:112-127"


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30x8_HighAccuracy(A30x8):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x8_Triton(A30x8):
    input_dtype = "fp16"
    input_format = "dhwc8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/brats/brats_npy/fp16_dhwc8"
    gpu_batch_size = 2
    offline_expected_qps = 230
    use_triton = True
    output_pinned_memory = False


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30x8_HighAccuracy_Triton(A30x8_Triton):
    pass


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class AGX_Xavier(OfflineGPUBaseConfig):
    system = System("AGX_Xavier", Architecture.Xavier, 1, cpu_arch=CPUArch.aarch64)
    gpu_batch_size = 1
    offline_expected_qps = 3
    use_direct_host_access = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class AGX_Xavier_HighAccuracy(AGX_Xavier):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class AGX_Xavier_Triton(AGX_Xavier):
    offline_expected_qps = 3
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class AGX_Xavier_HighAccuracy_Triton(AGX_Xavier_Triton):
    pass


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class AGX_Xavier_MaxQ(AGX_Xavier):
    offline_expected_qps = 2.1

    # power settings
    xavier_gpu_freq = 1032750000
    xavier_dla_freq = 115200000
    xavier_cpu_freq = 1190400
    xavier_emc_freq = 1331200000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class AGX_Xavier_HighAccuracy_MaxQ(AGX_Xavier_MaxQ):
    pass


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class Xavier_NX(OfflineGPUBaseConfig):
    system = System("Xavier_NX", Architecture.Xavier, 1, cpu_arch=CPUArch.aarch64)
    input_dtype = "fp16"
    input_format = "dhwc8"
    tensor_path = "${PREPROCESSED_DATA_DIR}/brats/brats_npy/fp16_dhwc8"
    workspace_size = 1073741824
    gpu_batch_size = 1
    gpu_copy_streams = 1
    offline_expected_qps = 1.5
    use_direct_host_access = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class Xavier_NX_HighAccuracy(Xavier_NX):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class Xavier_NX_Triton(Xavier_NX):
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class Xavier_NX_HighAccuracy_Triton(Xavier_NX_Triton):
    pass


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class Xavier_NX_MaxQ(Xavier_NX):
    # power settings
    xavier_gpu_freq = 752250000
    xavier_dla_freq = 115200000
    xavier_cpu_freq = 1190400
    xavier_emc_freq = 1600000000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class Xavier_NX_HighAccuracy_MaxQ(Xavier_NX_MaxQ):
    pass


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x1(OfflineGPUBaseConfig):
    system = System("T4", Architecture.Turing, 1)
    gpu_batch_size = 2
    offline_expected_qps = 8


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class T4x1_HighAccuracy(T4x1):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x1_Triton(T4x1):
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class T4x1_HighAccuracy_Triton(T4x1_Triton):
    pass


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x20(T4x1):
    system = System("T4", Architecture.Turing, 20)
    offline_expected_qps = 160


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class T4x20_HighAccuracy(T4x20):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x20_Triton(T4x20):
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class T4x20_HighAccuracy_Triton(T4x20_Triton):
    pass


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x8(T4x1):
    system = System("T4", Architecture.Turing, 8)
    offline_expected_qps = 64


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class T4x8_HighAccuracy(T4x8):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x8_Triton(T4x8):
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class T4x8_HighAccuracy_Triton(T4x8_Triton):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class Triton_CPU_2S_6258Rx1(OfflineCPUBaseConfig):
    system = System("Triton_CPU_2S_6258R", Architecture.Intel_CPU_x86_64, 1)
    precision = "fp32"
    offline_expected_qps = 4
    batch_size = 0
    input_dtype = "fp32"
    max_queue_delay_usec = 100
    model_name = "3dunet_int8_openvino"
    num_instances = 16
    ov_parameters = {
        'CPU_THREADS_NUM': '56',
        'CPU_THROUGHPUT_STREAMS': '8',
        'ENABLE_BATCH_PADDING': 'NO',
        'SKIP_OV_DYNAMIC_BATCHSIZE': 'YES'
    }
    tensor_path = "${PREPROCESSED_DATA_DIR}/brats/brats_npy/fp32"
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class Triton_CPU_2S_6258Rx1_HighAccuracy(Triton_CPU_2S_6258Rx1):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class Triton_CPU_4S_8380Hx1(OfflineCPUBaseConfig):
    system = System("Triton_CPU_4S_8380H", Architecture.Intel_CPU_x86_64, 1)
    precision = "fp32"
    offline_expected_qps = 10
    batch_size = 0
    input_dtype = "fp32"
    max_queue_delay_usec = 100
    model_name = "3dunet_int8_openvino"
    num_instances = 32
    ov_parameters = {
        'CPU_THREADS_NUM': '112',
        'CPU_THROUGHPUT_STREAMS': '16',
        'ENABLE_BATCH_PADDING': 'NO',
        'SKIP_OV_DYNAMIC_BATCHSIZE': 'YES'
    }
    tensor_path = "${PREPROCESSED_DATA_DIR}/brats/brats_npy/fp32"
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class Triton_CPU_4S_8380Hx1_HighAccuracy(Triton_CPU_4S_8380Hx1):
    pass
