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

from __future__ import annotations
from dataclasses import dataclass, asdict
from enum import Enum, unique
from typing import Any, Dict, Final, Optional, Union, Tuple

import re
import tensorrt as trt


__doc__ = """Stores constants and Enums related to MLPerf Inference"""


VERSION: Final[str] = "v1.1"
"""str: Current version of MLPerf Inference"""


TRT_LOGGER: Final[trt.Logger] = trt.Logger(trt.Logger.INFO)
"""trt.Logger: TRT Logger Singleton. TRT Runtime requires its logger to be a singleton object, so if we create multiple
runtimes with different trt.Logger objects, a segmentation fault will occur with 'The logger passed into
createInferRuntime differs from one already assigned'"""


@dataclass(eq=False, frozen=True)
class AliasedName:
    """
    Represents a name that has given aliases that are considered equivalent to the original name.
    """

    name: Optional[str]
    """Optional[str]: The main name this AliasedName is referred to as. None is reserved as a special value."""

    aliases: Tuple[str, ...] = tuple()
    """Tuple[str, ...]: A tuple of aliases that are considered equivalent, and should map to the main name."""

    patterns: Tuple[re.Pattern, ...] = tuple()
    """Tuple[re.Pattern, ...]: A tuple of regex patterns this AliasedName can match with. These have lower precendence
    than aliases and will only be checked if aliases has been exhausted without a match. Patterns are only used if
    the object to match is an instance of str."""

    def __str__(self):
        return self.name

    def __hash__(self):
        # Needs to be implemented since we specify eq=False. See Python dataclass documentation.
        if self.name is None:
            return hash(None)
        return hash(self.name.lower())

    def __eq__(self, other: Union[AliasedName, str, None]) -> bool:
        """
        Case insensitive equality check. Can be compared with another AliasedName or a str.

        If other is an AliasedName, returns True if self.name is case-insensitive equivalent to other.name.

        If other is a str, returns True if other is case-insensitive equivalent to self.name or any of the elements of
        self.aliases, or if it is a full match of any regex patterns in self.patterns.

        Args:
            other (Union[AliasedName, str, None]):
                The object to compare to

        Returns:
            bool: True if other is considered equal by the above rules. False otherwise, or if other is of an
            unrecognized type.
        """

        if other is None:
            return self.name is None

        if isinstance(other, AliasedName):
            if self.name is None or other.name is None:
                return self.name == other.name
            return self.name.lower() == other.name.lower()
        elif isinstance(other, str):
            if self.name is None:
                return False
            other_lower = other.lower()
            if self.name.lower() == other_lower or other_lower in (x.lower() for x in self.aliases):
                return True

            # No aliases matched, iterate through each regex.
            for pattern in self.patterns:
                if pattern.fullmatch(other):
                    return True
            return False
        else:
            return False


class AliasedNameEnum(Enum):
    """
    Meant to be used as a parent class for any Enum that has AliasedName values..
    """

    @classmethod
    def as_aliased_names(cls) -> List[AliasedName]:
        return [elem.value for elem in cls]

    @classmethod
    def as_strings(cls) -> List[str]:
        return list(map(str, cls.as_aliased_names()))

    @classmethod
    def get_match(cls, name: Union[AliasedName, str]) -> Optional[AliasedName]:
        """
        Attempts to return the element of this enum that is equivalent to `name`.

        Args:
            name (Union[AliasedName, str]):
                The name of an element we want

        Returns:
            Optional[AliasedName]: The AliasedName if found, None otherwise
        """
        for elem in cls:
            if elem.value == name:  # candidate must be on the left to use __eq__
                return elem
        return None

    def __eq__(self, other: Any) -> bool:
        """
        __eq__ override for members. Will compare directly if `other` is of the same __class__. Otherwise will attempt
        to use the __eq__ of the value.

        Args:
            other (Any):
                The object to compare to

        Returns:
            bool: True if other is equivalent to self directly, or self.value. False otherwise.
        """
        if self.__class__ is other.__class__:
            return self is other
        else:
            return self.value == other

    def __hash__(self) -> int:
        return hash(self.value)


@unique
class Benchmark(AliasedNameEnum):
    """Names of supported Benchmarks in MLPerf Inference."""

    BERT: AliasedName = AliasedName("bert")
    DLRM: AliasedName = AliasedName("dlrm")
    RNNT: AliasedName = AliasedName("rnnt", ("rnn-t",))
    ResNet50: AliasedName = AliasedName("resnet50", ("resnet",))
    SSDMobileNet: AliasedName = AliasedName("ssd-mobilenet", ("ssdmobilenet", "ssd-small"))
    SSDResNet34: AliasedName = AliasedName("ssd-resnet34", ("ssdresnet34", "ssd-large"))
    UNET3D: AliasedName = AliasedName("3d-unet", ("3dunet", "unet"))


@unique
class Scenario(AliasedNameEnum):
    """Names of supported workload scenarios in MLPerf Inference."""

    Offline: AliasedName = AliasedName("Offline")
    Server: AliasedName = AliasedName("Server")
    SingleStream: AliasedName = AliasedName("SingleStream", ("single-stream", "single_stream"))


@unique
class Action(AliasedNameEnum):
    """Names of actions performed by our MLPerf Inference pipeline."""

    GenerateConfFiles: AliasedName = AliasedName("generate_conf_files")
    GenerateEngines: AliasedName = AliasedName("generate_engines")
    Calibrate: AliasedName = AliasedName("calibrate")
    RunHarness: AliasedName = AliasedName("run_harness")
    RunCPUHarness: AliasedName = AliasedName("run_cpu_harness")
    RunAuditHarness: AliasedName = AliasedName("run_audit_harness")
    RunCPUAuditHarness: AliasedName = AliasedName("run_cpu_audit_harness")
    RunAuditVerify: AliasedName = AliasedName("run_audit_verification")
    RunCPUAuditVerify: AliasedName = AliasedName("run_cpu_audit_verification")


@unique
class AdHocSystemClassification(AliasedNameEnum):
    """Contains AliasedNames of useful classifications of System types. Not guaranteed to contain all known SystemClass
    types. Elements are also not guaranteed to actually match known system IDs.
    """

    Xavier: AliasedName = AliasedName("xavier", ("AGX_Xavier", "Xavier_NX"))
    StartFromDeviceEnabled: AliasedName = AliasedName(
        "start_from_device_enabled",
        ("A100-SXM-80GBx1",
         "A100-SXM-80GBx8",
         "A100-SXM-80GB-MIG_1x1g.10gb",
         "A100-SXM-80GB-MIG_28x1g.10gb",
         "A100-SXM-80GB-MIG_56x1g.10gb",
         "A100-SXM4-40GBx1",
         "A100-SXM4-40GBx8",
         "A100-SXM4-40GB-MIG_1x1g.5gb"))
    EndOnDeviceEnabled: AliasedName = AliasedName(
        "end_on_device_enabled",
        ("A100-SXM-80GBx1",
         "A100-SXM-80GBx8",
         "A100-SXM-80GB-MIG_1x1g.10gb",
         "A100-SXM-80GB-MIG_28x1g.10gb",
         "A100-SXM-80GB-MIG_56x1g.10gb",
         "A100-SXM4-40GBx1",
         "A100-SXM4-40GBx8",
         "A100-SXM4-40GB-MIG_1x1g.5gb"))
    IntelOpenVino: AliasedName = AliasedName(
        "intel_openvino",
        ("Triton_CPU_2S_6258Rx1",
         "Triton_CPU_4S_8380Hx1"))
    GPUBased: AliasedName = AliasedName(
        "gpu_based",
        patterns=(re.compile(r"(?!Triton_CPU_).*"),))

    # CatchAll must be the last element, as it will be the last in the list to be checked.
    CatchAll: AliasedName = AliasedName("CATCH_ALL", patterns=(re.compile(r".*"),))


@unique
class Precision(AliasedNameEnum):
    """Different numeric precisions that can be used by benchmarks. Not all benchmarks can use all precisions."""

    INT8: AliasedName = AliasedName("int8")
    FP16: AliasedName = AliasedName("fp16")
    FP32: AliasedName = AliasedName("fp32")


@unique
class InputFormats(AliasedNameEnum):
    """Different input formats that can be used by benchmarks. Not all benchmarks can use all input formats."""
    Linear: AliasedName = AliasedName("linear")
    CHW4: AliasedName = AliasedName("chw4")
    DHWC8: AliasedName = AliasedName("dhwc8")
    CDHW32: AliasedName = AliasedName("cdhw32")


@unique
class HarnessType(AliasedNameEnum):
    """Possible harnesses a benchmark can use."""
    LWIS: AliasedName = AliasedName("lwis")
    Custom: AliasedName = AliasedName("custom")
    Triton: AliasedName = AliasedName("triton")
    HeteroMIG: AliasedName = AliasedName("hetero")


@unique
class AccuracyTarget(Enum):
    """Possible accuracy targets a benchmark must meet. Determined by MLPerf Inference committee."""
    k_99: float = .99
    k_99_9: float = .999


@unique
class PowerSetting(AliasedNameEnum):
    """Possible power settings the system can be set in when running a benchmark."""
    MaxP: AliasedName = AliasedName("MaxP")
    MaxQ: AliasedName = AliasedName("MaxQ")


@dataclass(frozen=True)
class WorkloadSetting:
    """
    Describes the various settings used when running a benchmark workload. These are usually for different use cases that
    MLPerf Inference allows (i.e. power submission), or running the same workload with different software (i.e. Triton).
    """
    harness_type: HarnessType = HarnessType.Custom
    """HarnessType: Harness to use for this workload. Default: HarnessType.Custom"""

    accuracy_target: AccuracyTarget = AccuracyTarget.k_99
    """AccuracyTarget: Accuracy target for the benchmark. Default: AccuracyTarget.k_99"""

    power_setting: PowerSetting = PowerSetting.MaxP
    """PowerSetting: Power setting for the system during this workload. Default: PowerSetting.MaxP"""

    def shortname(self) -> str:
        return f"{self.harness_type.value.name}_{self.accuracy_target.name}_{self.power_setting.value.name}"

    def as_dict(self) -> Dict[str, Any]:
        """
        Convenience wrapper around dataclasses.asdict to convert this WorkloadSetting to a dict().

        Returns:
            Dict[str, Any]: This WorkloadSetting as a dict
        """
        return asdict(self)


G_DEFAULT_HARNESS_TYPES: Dict[Benchmark, HarnessType] = {
    Benchmark.BERT: HarnessType.Custom,
    Benchmark.DLRM: HarnessType.Custom,
    Benchmark.RNNT: HarnessType.Custom,
    Benchmark.ResNet50: HarnessType.LWIS,
    Benchmark.SSDMobileNet: HarnessType.LWIS,
    Benchmark.SSDResNet34: HarnessType.LWIS,
    Benchmark.UNET3D: HarnessType.LWIS,
}
"""Dict[Benchmark, HarnessType]: Defines the default harnesses (non-Triton) used for each benchmark."""


def config_ver_to_workload_setting(benchmark: Benchmark, config_ver: str) -> WorkloadSetting:
    """This method is a temporary workaround to retain legacy behavior as the codebase is incrementally refactored to
    use the new Python-style BenchmarkConfiguration instead of the old config.json files.

    Converts a legacy 'config_ver' ID to a new-style WorkloadSetting.

    Args:
        benchmark (Benchmark):
            The benchmark that is being processed. Used to decide the HarnessType.
        config_ver (str):
            The old-style 'config_ver' ID

    Returns:
        WorkloadSetting: The equivalent WorkloadSetting for the benchmark/config_ver.
    """
    harness_type = G_DEFAULT_HARNESS_TYPES[benchmark]
    if "openvino" in config_ver or "triton" in config_ver:
        harness_type = HarnessType.Triton
    elif "hetero" in config_ver:
        harness_type = HarnessType.HeteroMIG

    accuracy_target = AccuracyTarget.k_99
    if "high_accuracy" in config_ver:
        accuracy_target = AccuracyTarget.k_99_9

    power_setting = PowerSetting.MaxP
    if "maxq" in config_ver:
        power_setting = PowerSetting.MaxQ

    return WorkloadSetting(harness_type=harness_type, accuracy_target=accuracy_target, power_setting=power_setting)
