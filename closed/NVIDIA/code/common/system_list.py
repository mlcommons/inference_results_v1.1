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

from enum import Enum, unique
from collections import OrderedDict
import re
import subprocess
import os
import platform

from code.common.constants import CPUArch

NVIDIA_SMI_GPU_REGEX = re.compile(r"GPU (\d+): ([\w\- ]+) \(UUID: (GPU-[0-9a-f\-]+)\)")
"""
re.Pattern: Regex to match nvidia-smi output for GPU information
            match(1) - GPU index
            match(2) - GPU name
            match(3) - GPU UUID
"""
NVIDIA_SMI_MIG_REGEX = re.compile(
    r"\s+MIG\s+(\d+)g.(\d+)gb\s+Device\s+(\d+):\s+\(UUID:\s+(MIG-[0-9a-f\-]+)\)"
)
"""
re.Pattern: Regex to match nvidia-smi output for MIG information
            match(1) - Number of GPCs in the MIG slice
            match(2) - Allocated video memory capacity in the MIG slice in GB
            match(3) - MIG device ID
            match(4) - MIG instance UUID
"""


@unique
class Architecture(Enum):
    Turing = "Turing"
    Xavier = "Xavier"
    Ampere = "Ampere"

    Intel_CPU_x86_64 = "Intel CPU x86_64"

    Unknown = "Unknown"


class MIGSlice(object):

    def __init__(self, num_gpcs, mem_gb, device_id=None, uuid=None):
        """
        Describes a MIG instance. If optional arguments are set, then this MIGSlice describes an active MIG instance. If
        optional arguments are not set, then this MIGSlice describes an uninstantiated, but supported MIG instance.

        Arguments:
            num_gpcs: Number of GPCs in this MIG slice
            mem_gb: Allocated video memory capacity in this MIG slice in GB

        Optional arguments:
            device_id: Device ID of the GPU this MIG is a part of
            uuid: UUID of this MIG instance
        """
        self.num_gpcs = num_gpcs
        self.mem_gb = mem_gb
        self.device_id = device_id
        self.uuid = uuid

        # One cannot be set without the other.
        assert (device_id is None) == (uuid is None)

    def __str__(self):
        return "{:d}g.{:d}gb".format(self.num_gpcs, self.mem_gb)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return str(self) == str(other)

    def is_active_slice(self):
        return self.device_id is None

    def get_gpu_uuid(self):
        return self.uuid.split("/")[0][4:]  # First 4 characters are "MIG-"

    def get_gpu_instance_id(self):
        return int(self.uuid.split("/")[1])

    def get_compute_instance_id(self):
        return int(self.uuid.split("/")[2])


class MIGConfiguration(object):

    def __init__(self, conf):
        """
        Stores information about a system's MIG configuration.

        conf: An OrderedDict of gpu_id -> { MIGSlice -> Count }
        """
        self.conf = conf

    def check_compatible(self, valid_mig_slices):
        """
        Given a list of valid MIGSlices, checks if this MIGConfiguration only contains MIGSlices that are described in
        the list.
        """
        m = {str(mig) for mig in valid_mig_slices}
        for gpu_id in self.conf:
            for mig in self.conf[gpu_id]:
                if str(mig) not in m:
                    return False
        return True

    def num_gpus(self):
        """
        Returns the number of GPUs with active MIG instances in this MIGConfiguration.
        """
        return len(self.conf)

    def num_mig_slices(self):
        """
        Returns the number of total active MIG instances across all GPUs in this MIGConfiguration
        """
        i = 0
        for gpu_id in self.conf:
            for mig in self.conf[gpu_id]:
                i += self.conf[gpu_id][mig]
        return i

    def __str__(self):
        """
        Returns a string that describes this MIG configuration.

        Examples:
          - For 1x 1-GPC: 1x1g.10gb
          - For 1x 1-GPC, 2x 2-GPC, and 3x 3-GPC: 1x1g.10gb_2x2g.20gb_1x3g.30gb
        """
        # Add up the slices on each GPU by MIGSlice
        flattened = OrderedDict()
        for gpu_id in self.conf:
            for mig in self.conf[gpu_id]:
                if mig not in flattened:
                    flattened[mig] = 0
                flattened[mig] += self.conf[gpu_id][mig]
        # Assert if 1) there's no MIG instance at all but this is called for, or
        #           2) MIG config is captured but no associated instance is found
        # as correct string cannot be returned for such cases
        assert all(flattened) and all([flattened[i] > 0 for i in flattened]),\
            f"Unsupported MIG configuration: {self.conf}"
        return "_".join(sorted(["{}x{}".format(flattened[mig], str(mig))]))

    @staticmethod
    def get_gpu_mig_slice_mapping():
        """
        Returns a dict containing mapping between gpu uuid and list of mig slices on that gpu.
        """
        p = subprocess.Popen("nvidia-smi -L", universal_newlines=True, shell=True, stdout=subprocess.PIPE)
        gpu_mig_slice_mapping = dict()
        for line in p.stdout:
            gpu_match = NVIDIA_SMI_GPU_REGEX.match(line)
            if gpu_match is not None:
                gpu_uuid = gpu_match.group(3)
                gpu_mig_slice_mapping[gpu_uuid] = []

            mig_match = NVIDIA_SMI_MIG_REGEX.match(line)
            if mig_match is not None:
                gpu_mig_slice_mapping[gpu_uuid].append(MIGSlice(
                    int(mig_match.group(1)),  # num_gpcs
                    int(mig_match.group(2)),  # mem_gb
                    int(mig_match.group(3)),  # device_id
                    mig_match.group(4)  # uuid
                ))
        return gpu_mig_slice_mapping

    @staticmethod
    def from_nvidia_smi():
        visible_gpu_instances = set()
        if os.environ.get("CUDA_VISIBLE_DEVICES"):
            for g in os.environ.get("CUDA_VISIBLE_DEVICES").split(","):
                if g.startswith("MIG"):
                    visible_gpu_instances.add(g)

        gpu_mig_slice_mapping = MIGConfiguration.get_gpu_mig_slice_mapping()
        conf = OrderedDict()
        for gpu, mig_slices in gpu_mig_slice_mapping.items():
            conf[gpu] = {}
            for mig_slice in mig_slices:
                if (not os.environ.get("CUDA_VISIBLE_DEVICES")) or (
                    mig_slice.uuid in visible_gpu_instances
                ):
                    if mig_slice not in conf[gpu]:
                        conf[gpu][mig_slice] = 0
                    conf[gpu][mig_slice] += 1
        return MIGConfiguration(conf)


class System(object):
    """
    System class contains information on the GPUs used in our submission systems.

    gpu: ID of the GPU being used
    pci_id: PCI ID of the GPU
    arch: Architecture of the accelerator
    count: Number of GPUs used on the system
    mig_conf: MIG configuration (if applicable)
    """

    def __init__(self, gpu, arch, count, pci_id=None, mig_conf=None, cpu_arch=CPUArch.x86_64):
        self.gpu = gpu
        self.arch = arch
        self.count = count
        self.mig_conf = mig_conf
        self.pci_id = pci_id
        self.uses_mig = mig_conf is not None
        self.cpu_arch = cpu_arch

    def get_id(self):
        sid = str(self.gpu)
        if self.arch != Architecture.Xavier:
            # Append cpu_arch only to non-x86 systems.
            if self.cpu_arch != CPUArch.x86_64:
                sid += "_" + self.cpu_arch.valstr()
            sid += f"x{self.count}"
        if self.mig_conf is not None:
            sid = "{:}-MIG_{:}".format(self.gpu, str(self.mig_conf))
        return sid

    def __str__(self):
        return self.get_id()

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return str(self) == str(other)


class SystemClass(object):
    def __init__(self, gpu, aliases, pci_ids, arch, supported_counts, valid_mig_slices=None, cpu_arch=CPUArch.x86_64):
        """
        SystemClass describes classes of submissions systems with different variations. SystemClass objects are
        hardcoded as supported systems and must be defined in KnownSystems below to be recognized a valid system for the
        pipeline.

        Args:
            gpu: ID of the GPU being used, usually the name reported by nvidia-smi
            aliases: Different names of cards reported by nvidia-smi that use the same SKUs, i.e. Titan RTX and Quadro
                     RTX 8000
            pci_ids: PCI IDs of cards that match this system configuration that use the same SKUs
            arch: Architecture of the accelerator
            supported_counts: Counts of GPUs for supported multi-GPU systems, i.e. [1, 2, 4] to support 1x, 2x, and 4x
                              GPU systems
            valid_mig_slices: List of supported MIGSlices. None if MIG not supported.
        """
        self.gpu = gpu
        self.aliases = aliases
        self.pci_ids = pci_ids
        self.arch = arch
        self.supported_counts = supported_counts
        self.valid_mig_slices = valid_mig_slices
        self.supports_mig = valid_mig_slices is not None
        self.cpu_arch = cpu_arch

    def __str__(self):
        return "SystemClass(gpu={}, aliases={}, pci_ids={}, arch={}, counts={}, cpu_arch={})".format(
            self.gpu,
            self.aliases,
            self.pci_ids,
            self.arch,
            self.supported_counts,
            self.cpu_arch.value.name)

    def get_match(self, name, count, pci_id=None, mig_conf=None):
        """
        Attempts to match a certain GPU configuration with this SystemClass. If the configuration does not match,
        returns None. Otherwise, returns a System object with metadata about the configuration.

        mig_conf should be a MIGConfiguration object.
        """
        # PCI ID has precedence over name, as pre-release chips often are not named yet in nvidia-smi
        gpu_match = False
        if pci_id is not None and len(self.pci_ids) > 0:
            gpu_match = pci_id in self.pci_ids
        # Attempt to match a name if PCI ID is not specified, or if the system has no known PCI IDs
        # This is an else block, but we explicitly show condition for clarity
        elif pci_id is None or len(self.pci_ids) == 0:
            gpu_match = name in self.aliases

        if not gpu_match:
            return None

        # If GPU matches, match the count and mig configs (if applicable)
        if count not in self.supported_counts:
            return None

        if self.supports_mig and mig_conf is not None and not mig_conf.check_compatible(self.valid_mig_slices):
            return None

        if CPUArch.get_match(platform.processor()) != self.cpu_arch:
            return None

        return System(self.gpu, self.arch, count, pci_id=pci_id, mig_conf=mig_conf, cpu_arch=self.cpu_arch)


class KnownSystems(object):
    """
    Global List of supported systems
    """

    A100_PCIe_40GB = SystemClass("A100-PCIe", ["A100-PCIE-40GB"], ["20F1", "20BF"], Architecture.Ampere, [1, 2, 8])
    A100_PCIe_40GB_aarch64 = SystemClass("A100-PCIe", ["A100-PCIE-40GB"], ["20F1", "20BF"], Architecture.Ampere, [1, 2, 4], cpu_arch=CPUArch.aarch64)
    A100_PCIe_80GB = SystemClass("A100-PCIe-80GB", ["A100-PCIE-80GB"], ["20B5"], Architecture.Ampere, [1, 2, 8])
    A100_PCIe_80GB_aarch64 = SystemClass("A100-PCIe-80GB", ["A100-PCIE-80GB"], ["20B5"], Architecture.Ampere, [1, 2, 4], cpu_arch=CPUArch.aarch64)
    A100_SXM4_40GB = SystemClass("A100-SXM4-40GB", ["A100-SXM4-40GB"], ["20B0"], Architecture.Ampere, [1, 8],
                                 valid_mig_slices=[MIGSlice(1, 5), MIGSlice(2, 10), MIGSlice(3, 20)])
    A100_SXM_80GB = SystemClass("A100-SXM-80GB", ["A100-SXM-80GB"], ["20B2"], Architecture.Ampere, [1, 4, 8],
                                valid_mig_slices=[MIGSlice(1, 10), MIGSlice(2, 20), MIGSlice(3, 40)])
    GeForceRTX_3080 = SystemClass("GeForceRTX3080", ["GeForce RTX 3080"], ["2206"], Architecture.Ampere, [1])
    GeForceRTX_3090 = SystemClass("GeForceRTX3090", ["GeForce RTX 3090", "Quadro RTX A6000", "RTX A6000"],
                                  ["2204", "2230"], Architecture.Ampere, [1])
    A10 = SystemClass("A10", ["A10"], ["2236"], Architecture.Ampere, [1, 8])
    T4 = SystemClass("T4", ["Tesla T4", "T4 32GB"], ["1EB8", "1EB9"], Architecture.Turing, [1, 8, 20])
    AGX_Xavier = SystemClass("AGX_Xavier", ["Jetson-AGX"], [], Architecture.Xavier, [1], cpu_arch=CPUArch.aarch64)
    Xavier_NX = SystemClass("Xavier_NX", ["Xavier NX"], [], Architecture.Xavier, [1], cpu_arch=CPUArch.aarch64)
    A30 = SystemClass("A30", ["A30"], ["20B7"], Architecture.Ampere, [1, 8],
                      valid_mig_slices=[MIGSlice(1, 6), MIGSlice(2, 12), MIGSlice(4, 24)])

    # CPU Systems
    Triton_CPU_2S_6258R = SystemClass("Triton_CPU_2S_6258R", ["2S_6258R"], [], Architecture.Intel_CPU_x86_64, [1])
    Triton_CPU_4S_8380H = SystemClass("Triton_CPU_4S_8380H", ["4S_8380H"], [], Architecture.Intel_CPU_x86_64, [1])

    @staticmethod
    def get_all_system_classes():
        return [
            getattr(KnownSystems, attr)
            for attr in dir(KnownSystems)
            if type(getattr(KnownSystems, attr)) == SystemClass
        ]

    @staticmethod
    def get_all_systems():
        all_classes = KnownSystems.get_all_system_classes()
        all_systems = []
        for system_class in all_classes:
            for count in system_class.supported_counts:
                all_systems.append(System(system_class.gpu, system_class.arch, count, cpu_arch=system_class.cpu_arch))
                if count == 1 and system_class.valid_mig_slices is not None:
                    for mig_slice in system_class.valid_mig_slices:
                        conf = {"DummyGPU": {mig_slice: 1}}
                        mig_conf = MIGConfiguration(conf)
                        all_systems.append(System(system_class.gpu, system_class.arch, count, mig_conf=mig_conf))

        # Special handling for 56 MIG DGX-A100
        all_systems.append(
            System("A100-SXM-80GB", Architecture.Ampere, 8,
                   mig_conf=MIGConfiguration({"DummyGPU": {MIGSlice(1, 10): 56}}))
        )
        # Special handling for 32 MIG A30
        all_systems.append(
            System("A30", Architecture.Ampere, 8,
                   mig_conf=MIGConfiguration({"DummyGPU": {MIGSlice(1, 6): 32}}))
        )

        return all_systems
