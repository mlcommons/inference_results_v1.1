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

from code.common import logging, run_command, is_xavier, is_xavier_agx, is_xavier_nx
import subprocess
from typing import List, Optional, Union, Dict, Any
from dataclasses import dataclass


@dataclass
class XavierPowerState:
    gpu_freq: int
    dla_freq: int
    cpu_freq: int
    emc_freq: int


@dataclass
class ServerPowerState:
    power_limit: Union[int, List[int], None]
    cpu_freq: Optional[int]


PowerState = Union[XavierPowerState, ServerPowerState]


def extract_field(main_args: Dict[str, Any], benchmark_conf: Dict[str, Any], field_name: str) -> Any:
    """Extracts a field from the given parameters, preferring main_args if it is supplied."""
    field = main_args.get(field_name, None)
    if field is None:
        field = benchmark_conf.get(field_name, None)
    return field


def get_power_state_server(main_args: Dict[str, Any], benchmark_conf: Dict[str, Any]):
    # override bencmark conf if arg was also supplied to main
    power_limit = benchmark_conf["power_limit"] = extract_field(main_args, benchmark_conf, "power_limit")
    cpu_freq = benchmark_conf["cpu_freq"] = extract_field(main_args, benchmark_conf, "cpu_freq")

    if power_limit or cpu_freq:
        return ServerPowerState(power_limit, cpu_freq)

    return None


def get_power_state_xavier(main_args: Dict[str, Any], benchmark_conf: Dict[str, Any]):
    # override bencmark conf if arg was also supplied to main
    gpu_freq = benchmark_conf["xavier_gpu_freq"] = extract_field(main_args, benchmark_conf, "xavier_gpu_freq")
    dla_freq = benchmark_conf["xavier_dla_freq"] = extract_field(main_args, benchmark_conf, "xavier_dla_freq")
    cpu_freq = benchmark_conf["xavier_cpu_freq"] = extract_field(main_args, benchmark_conf, "xavier_cpu_freq")
    emc_freq = benchmark_conf["xavier_emc_freq"] = extract_field(main_args, benchmark_conf, "xavier_emc_freq")

    # if any of these flags are set, all of them should be
    frequencies = [gpu_freq, dla_freq, cpu_freq, emc_freq]
    if None in frequencies:
        assert not any(frequencies), f"All frequencies must be set ({frequencies})"
        return None
    return XavierPowerState(*frequencies)


def get_power_state(main_args: Dict[str, Any], benchmark_conf: Dict[str, Any]):
    """Parse args and get target power state"""

    if benchmark_conf["use_cpu"]:
        return None

    if is_xavier():
        return get_power_state_xavier(main_args, benchmark_conf)
    else:
        return get_power_state_server(main_args, benchmark_conf)


def set_cpufreq(cpu_freq: int) -> List[float]:
    # Record current cpu governor
    cmd = "sudo cpupower -c all frequency-set -g userspace"
    logging.info(f"Set cpu power governor: userspace")
    run_command(cmd)

    # Set cpu freq
    cmd = f"sudo cpupower -c all frequency-set -f {cpu_freq}"
    logging.info(f"Setting cpu frequency: {cmd}")
    run_command(cmd)


def reset_cpufreq():
    # Record current cpu governor
    cmd = "sudo cpupower -c all frequency-set -g ondemand"
    logging.info(f"Set cpu power governor: ondemand")
    run_command(cmd)


def set_power_state_server(power_state: ServerPowerState) -> List[float]:
    """Record the current power limit and set power limit using nvidia-smi."""

    # Record current power limits.
    if power_state.power_limit:
        cmd = "nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits"
        logging.info(f"Getting current GPU power limits: {cmd}")
        output = run_command(cmd, get_output=True, tee=False)
        current_limits = [float(line) for line in output]

        # Set power limit to the specified value.
        cmd = f"sudo nvidia-smi -pl {power_state.power_limit}"
        logging.info(f"Setting current GPU power limits: {cmd}")
        run_command(cmd)

    if power_state.cpu_freq:
        set_cpufreq(power_state.cpu_freq)

    return ServerPowerState(current_limits, None)


def reset_power_state_server(power_state: ServerPowerState) -> None:
    """Record the current power limit and set power limit using nvidia-smi."""

    # Reset power limit to the specified value.
    power_limits = power_state.power_limit
    for i in range(len(power_limits)):
        cmd = f"sudo nvidia-smi -i {i} -pl {power_limits[i]}"
        logging.info(f"Resetting power limit for GPU {i}: {cmd}")
        run_command(cmd)


nvpmodel_template = {"xavier_agx": """
< PARAM TYPE=FILE NAME=CPU_ONLINE >
CORE_0 /sys/devices/system/cpu/cpu0/online
CORE_1 /sys/devices/system/cpu/cpu1/online
CORE_2 /sys/devices/system/cpu/cpu2/online
CORE_3 /sys/devices/system/cpu/cpu3/online
CORE_4 /sys/devices/system/cpu/cpu4/online
CORE_5 /sys/devices/system/cpu/cpu5/online
CORE_6 /sys/devices/system/cpu/cpu6/online
CORE_7 /sys/devices/system/cpu/cpu7/online

< PARAM TYPE=FILE NAME=TPC_POWER_GATING >
TPC_PG_MASK /sys/devices/gpu.0/tpc_pg_mask

< PARAM TYPE=FILE NAME=GPU_POWER_CONTROL_ENABLE >
GPU_PWR_CNTL_EN /sys/devices/gpu.0/power/control

< PARAM TYPE=FILE NAME=GPU_POWER_CONTROL_DISABLE >
GPU_PWR_CNTL_DIS /sys/devices/gpu.0/power/control

< PARAM TYPE=CLOCK NAME=CPU_DENVER_0 >
FREQ_TABLE /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies
MAX_FREQ /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq
MIN_FREQ /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq

< PARAM TYPE=CLOCK NAME=CPU_DENVER_1 >
FREQ_TABLE /sys/devices/system/cpu/cpu2/cpufreq/scaling_available_frequencies
MAX_FREQ /sys/devices/system/cpu/cpu2/cpufreq/scaling_max_freq
MIN_FREQ /sys/devices/system/cpu/cpu2/cpufreq/scaling_min_freq

< PARAM TYPE=CLOCK NAME=CPU_DENVER_2 >
FREQ_TABLE /sys/devices/system/cpu/cpu4/cpufreq/scaling_available_frequencies
MAX_FREQ /sys/devices/system/cpu/cpu4/cpufreq/scaling_max_freq
MIN_FREQ /sys/devices/system/cpu/cpu4/cpufreq/scaling_min_freq

< PARAM TYPE=CLOCK NAME=CPU_DENVER_3 >
FREQ_TABLE /sys/devices/system/cpu/cpu6/cpufreq/scaling_available_frequencies
MAX_FREQ /sys/devices/system/cpu/cpu6/cpufreq/scaling_max_freq
MIN_FREQ /sys/devices/system/cpu/cpu6/cpufreq/scaling_min_freq

< PARAM TYPE=CLOCK NAME=GPU >
FREQ_TABLE /sys/devices/17000000.gv11b/devfreq/17000000.gv11b/available_frequencies
MAX_FREQ /sys/devices/17000000.gv11b/devfreq/17000000.gv11b/max_freq
MIN_FREQ /sys/devices/17000000.gv11b/devfreq/17000000.gv11b/min_freq

< PARAM TYPE=CLOCK NAME=EMC >
MAX_FREQ /sys/kernel/nvpmodel_emc_cap/emc_iso_cap

< PARAM TYPE=CLOCK NAME=DLA_CORE >
MAX_FREQ /sys/kernel/nvpmodel_emc_cap/nafll_dla
MAX_FREQ_KNEXT /sys/kernel/nvpmodel_emc_cap/nafll_dla

< PARAM TYPE=CLOCK NAME=DLA_FALCON >
MAX_FREQ /sys/kernel/nvpmodel_emc_cap/nafll_dla_falcon
MAX_FREQ_KNEXT /sys/kernel/nvpmodel_emc_cap/nafll_dla_falcon

< PARAM TYPE=CLOCK NAME=PVA_VPS >
MAX_FREQ /sys/kernel/nvpmodel_emc_cap/nafll_pva_vps
MAX_FREQ_KNEXT /sys/kernel/nvpmodel_emc_cap/nafll_pva_vps

< PARAM TYPE=CLOCK NAME=PVA_CORE >
MAX_FREQ /sys/kernel/nvpmodel_emc_cap/nafll_pva_core
MAX_FREQ_KNEXT /sys/kernel/nvpmodel_emc_cap/nafll_pva_core

< PARAM TYPE=CLOCK NAME=CVNAS >
MAX_FREQ /sys/kernel/nvpmodel_emc_cap/nafll_cvnas
MAX_FREQ_KNEXT /sys/kernel/nvpmodel_emc_cap/nafll_cvnas

< POWER_MODEL ID=0 NAME=MAXN >
CPU_ONLINE CORE_0 1
CPU_ONLINE CORE_1 1
CPU_ONLINE CORE_2 1
CPU_ONLINE CORE_3 1
CPU_ONLINE CORE_4 1
CPU_ONLINE CORE_5 1
CPU_ONLINE CORE_6 1
CPU_ONLINE CORE_7 1
TPC_POWER_GATING TPC_PG_MASK 0
GPU_POWER_CONTROL_ENABLE GPU_PWR_CNTL_EN on
CPU_DENVER_0 MIN_FREQ 1200000
CPU_DENVER_0 MAX_FREQ -1
CPU_DENVER_1 MIN_FREQ 1200000
CPU_DENVER_1 MAX_FREQ -1
CPU_DENVER_2 MIN_FREQ 1200000
CPU_DENVER_2 MAX_FREQ -1
CPU_DENVER_3 MIN_FREQ 1200000
CPU_DENVER_3 MAX_FREQ -1
GPU MIN_FREQ 318750000
GPU MAX_FREQ -1
GPU_POWER_CONTROL_DISABLE GPU_PWR_CNTL_DIS auto
EMC MAX_FREQ 0
DLA_CORE MAX_FREQ -1
DLA_FALCON MAX_FREQ -1
PVA_VPS MAX_FREQ -1
PVA_CORE MAX_FREQ -1
CVNAS MAX_FREQ -1

< POWER_MODEL ID=8 NAME=MODE_MLPERF_V1_MAXQ >
CPU_ONLINE CORE_0 1
CPU_ONLINE CORE_1 1
CPU_ONLINE CORE_2 0
CPU_ONLINE CORE_3 0
CPU_ONLINE CORE_4 0
CPU_ONLINE CORE_5 0
CPU_ONLINE CORE_6 0
CPU_ONLINE CORE_7 0
TPC_POWER_GATING TPC_PG_MASK 0
GPU_POWER_CONTROL_ENABLE GPU_PWR_CNTL_EN on
CPU_DENVER_0 MIN_FREQ {cpu_clock}
CPU_DENVER_0 MAX_FREQ {cpu_clock}
GPU MIN_FREQ 318750000
GPU MAX_FREQ {gpu_clock}
GPU_POWER_CONTROL_DISABLE GPU_PWR_CNTL_DIS auto
EMC MAX_FREQ {emc_clock}
DLA_CORE MAX_FREQ {dla_clock}
DLA_FALCON MAX_FREQ 630000000
PVA_VPS MAX_FREQ 760000000
PVA_CORE MAX_FREQ 532000000
CVNAS MAX_FREQ 1011200000

< PM_CONFIG DEFAULT=8 >
< FAN_CONFIG DEFAULT=cool >

""",
                     "xavier_nx": """
< PARAM TYPE=FILE NAME=CPU_ONLINE >
CORE_0 /sys/devices/system/cpu/cpu0/online
CORE_1 /sys/devices/system/cpu/cpu1/online
CORE_2 /sys/devices/system/cpu/cpu2/online
CORE_3 /sys/devices/system/cpu/cpu3/online
CORE_4 /sys/devices/system/cpu/cpu4/online
CORE_5 /sys/devices/system/cpu/cpu5/online

< PARAM TYPE=FILE NAME=TPC_POWER_GATING >
TPC_PG_MASK /sys/devices/gpu.0/tpc_pg_mask

< PARAM TYPE=FILE NAME=GPU_POWER_CONTROL_ENABLE >
GPU_PWR_CNTL_EN /sys/devices/gpu.0/power/control

< PARAM TYPE=FILE NAME=GPU_POWER_CONTROL_DISABLE >
GPU_PWR_CNTL_DIS /sys/devices/gpu.0/power/control

< PARAM TYPE=CLOCK NAME=CPU_DENVER_0 >
FREQ_TABLE /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies
MAX_FREQ /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq
MIN_FREQ /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq

< PARAM TYPE=CLOCK NAME=CPU_DENVER_1 >
FREQ_TABLE /sys/devices/system/cpu/cpu2/cpufreq/scaling_available_frequencies
MAX_FREQ /sys/devices/system/cpu/cpu2/cpufreq/scaling_max_freq
MIN_FREQ /sys/devices/system/cpu/cpu2/cpufreq/scaling_min_freq

< PARAM TYPE=CLOCK NAME=CPU_DENVER_2 >
FREQ_TABLE /sys/devices/system/cpu/cpu4/cpufreq/scaling_available_frequencies
MAX_FREQ /sys/devices/system/cpu/cpu4/cpufreq/scaling_max_freq
MIN_FREQ /sys/devices/system/cpu/cpu4/cpufreq/scaling_min_freq

< PARAM TYPE=CLOCK NAME=GPU >
FREQ_TABLE /sys/devices/17000000.gv11b/devfreq/17000000.gv11b/available_frequencies
MAX_FREQ /sys/devices/17000000.gv11b/devfreq/17000000.gv11b/max_freq
MIN_FREQ /sys/devices/17000000.gv11b/devfreq/17000000.gv11b/min_freq

< PARAM TYPE=CLOCK NAME=EMC >
MAX_FREQ /sys/kernel/nvpmodel_emc_cap/emc_iso_cap

< PARAM TYPE=CLOCK NAME=DLA_CORE >
MAX_FREQ /sys/kernel/nvpmodel_emc_cap/nafll_dla
MAX_FREQ_KNEXT /sys/kernel/nvpmodel_emc_cap/nafll_dla

< PARAM TYPE=CLOCK NAME=DLA_FALCON >
MAX_FREQ /sys/kernel/nvpmodel_emc_cap/nafll_dla_falcon
MAX_FREQ_KNEXT /sys/kernel/nvpmodel_emc_cap/nafll_dla_falcon

< PARAM TYPE=CLOCK NAME=PVA_VPS >
MAX_FREQ /sys/kernel/nvpmodel_emc_cap/nafll_pva_vps
MAX_FREQ_KNEXT /sys/kernel/nvpmodel_emc_cap/nafll_pva_vps

< PARAM TYPE=CLOCK NAME=PVA_CORE >
MAX_FREQ /sys/kernel/nvpmodel_emc_cap/nafll_pva_core
MAX_FREQ_KNEXT /sys/kernel/nvpmodel_emc_cap/nafll_pva_core

< PARAM TYPE=CLOCK NAME=CVNAS >
MAX_FREQ /sys/kernel/nvpmodel_emc_cap/nafll_cvnas
MAX_FREQ_KNEXT /sys/kernel/nvpmodel_emc_cap/nafll_cvnas

< POWER_MODEL ID=0 NAME=MODE_15W_2CORE >
CPU_ONLINE CORE_0 1
CPU_ONLINE CORE_1 1
CPU_ONLINE CORE_2 0
CPU_ONLINE CORE_3 0
CPU_ONLINE CORE_4 0
CPU_ONLINE CORE_5 0
TPC_POWER_GATING TPC_PG_MASK 1
GPU_POWER_CONTROL_ENABLE GPU_PWR_CNTL_EN on
CPU_DENVER_0 MIN_FREQ 1190400
CPU_DENVER_0 MAX_FREQ 1907200
GPU MIN_FREQ 0
GPU MAX_FREQ 1109250000
GPU_POWER_CONTROL_DISABLE GPU_PWR_CNTL_DIS auto
EMC MAX_FREQ 1600000000
DLA_CORE MAX_FREQ 1100800000
DLA_FALCON MAX_FREQ 640000000
PVA_VPS MAX_FREQ 819200000
PVA_CORE MAX_FREQ 601600000
CVNAS MAX_FREQ 576000000

< POWER_MODEL ID=8 NAME=MODE_MLPERF_V1_MAXQ >
CPU_ONLINE CORE_0 1
CPU_ONLINE CORE_1 1
CPU_ONLINE CORE_2 0
CPU_ONLINE CORE_3 0
CPU_ONLINE CORE_4 0
CPU_ONLINE CORE_5 0
TPC_POWER_GATING TPC_PG_MASK 1
GPU_POWER_CONTROL_ENABLE GPU_PWR_CNTL_EN on
CPU_DENVER_0 MIN_FREQ {cpu_clock}
CPU_DENVER_0 MAX_FREQ {cpu_clock}
GPU MIN_FREQ 0
GPU MAX_FREQ {gpu_clock}
GPU_POWER_CONTROL_DISABLE GPU_PWR_CNTL_DIS auto
EMC MAX_FREQ {emc_clock}
DLA_CORE MAX_FREQ {dla_clock}
DLA_FALCON MAX_FREQ 640000000
PVA_VPS MAX_FREQ 819200000
PVA_CORE MAX_FREQ 601600000
CVNAS MAX_FREQ 576000000

< PM_CONFIG DEFAULT=8 >
< FAN_CONFIG DEFAULT=cool >

"""
                     }


def set_power_state_xavier(power_state: XavierPowerState) -> None:
    """Record the current power state and set power limit using nvpmodel."""

    # Set power limit to the specified value
    if is_xavier_agx():
        platform = "xavier_agx"
    elif is_xavier_nx():
        platform = "xavier_nx"
    else:
        raise RuntimeError("Xavier platform must be AGX or NX")

    with open("build/nvpmodel.temp.conf", "w") as f:
        f.write(nvpmodel_template[platform].format(gpu_clock=power_state.gpu_freq,
                                                   dla_clock=power_state.dla_freq,
                                                   cpu_clock=power_state.cpu_freq,
                                                   emc_clock=power_state.emc_freq))
    cmd = "sudo /usr/sbin/nvpmodel -f build/nvpmodel.temp.conf -m 8 && sudo /usr/sbin/nvpmodel -d cool"
    logging.info(f"Setting current nvpmodel conf: {cmd}")
    run_command(cmd)

    return None


def reset_power_state_xavier(power_limits: List[float]) -> None:
    """Reset power limit using nvpmodel conf"""

    # Reset power limit to the specified value.
    cmd = "sudo /usr/sbin/nvpmodel -m 0 && sudo /usr/sbin/nvpmodel -d cool"
    logging.info(f"Resetting nvpmodel conf: {cmd}")
    run_command(cmd)


class ScopedPowerLimit:
    """
        Create scope GPU power upper limit is overridden to the specified value.
        Setting power_limit to None to disable the scoped power limit.
    """

    def __init__(self, target_power_state: PowerState):
        self.target_power_state = target_power_state
        self.current_power_state = None
        self.set_power_limits = set_power_state_server if not is_xavier() else set_power_state_xavier
        self.reset_power_limits = reset_power_state_server if not is_xavier() else reset_power_state_xavier

    def __enter__(self):
        if self.target_power_state is not None:
            self.current_power_state = self.set_power_limits(self.target_power_state)

    def __exit__(self, type, value, traceback):
        if self.target_power_state is not None:
            self.reset_power_limits(self.current_power_state)
