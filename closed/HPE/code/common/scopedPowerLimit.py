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

from code.common import logging, run_command, is_xavier
import subprocess
from typing import List, Optional


def set_power_limits(power_limit: int) -> List[float]:
    """Record the current power limit and set power limit using nvidia-smi."""

    # Record current power limits.
    cmd = "nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits"
    logging.info(f"Getting current GPU power limits: {cmd}")
    output = run_command(cmd, get_output=True, tee=False)
    current_limits = [float(line) for line in output]

    # Set power limit to the specified value.
    cmd = f"sudo nvidia-smi -pl {power_limit}"
    logging.info(f"Setting current GPU power limits: {cmd}")
    run_command(cmd)

    return current_limits


def reset_power_limits(power_limits: List[float]) -> None:
    """Record the current power limit and set power limit using nvidia-smi."""

    # Reset power limit to the specified value.
    for i in range(len(power_limits)):
        cmd = f"sudo nvidia-smi -i {i} -pl {power_limits[i]}"
        logging.info(f"Resetting power limit for GPU {i}: {cmd}")
        run_command(cmd)


class ScopedPowerLimit:
    """
        Create scope GPU power upper limit is overridden to the specified value.
        Setting power_limit to None to disable the scoped power limit.
    """

    def __init__(self, power_limit: Optional[int] = None):
        # Xavier does not have nvidia-smi, so we cannot set power limit in this way.
        if is_xavier() and power_limit is not None:
            raise RuntimeError("Per-benchmark power limit setting is not supported on Xavier/NX!")
        self.power_limit = power_limit
        self.current_limits = None

    def __enter__(self):
        if self.power_limit is not None:
            self.current_limits = set_power_limits(self.power_limit)

    def __exit__(self, type, value, traceback):
        if self.power_limit is not None:
            reset_power_limits(self.current_limits)
