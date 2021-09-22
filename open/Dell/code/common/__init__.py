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

import json
import platform
import subprocess
import sys
import re

from glob import glob

# TODO: Remove when constants.py is integrated
VERSION = "v1.0"

import logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s %(filename)s:%(lineno)d %(levelname)s] %(message)s")

from code.common.system_list import KnownSystems, MIGConfiguration


def is_aarch64():
    return platform.processor() == "aarch64"


def is_xavier():
    if not is_aarch64():
        return False

    # Use the model file to determine whether the it's a Xavier system.
    return os.path.exists("/sys/firmware/devicetree/base/model")


def check_xavier_version(s):
    if not is_xavier():
        return False

    with open("/sys/firmware/devicetree/base/model", "r") as f:
        txt = f.read()

    return s in txt


def is_xavier_nx():
    return check_xavier_version("NX")


def is_xavier_agx():
    return check_xavier_version("AGX")


def check_mig_enabled():
    """Check if MIG is enabled on input GPU."""

    p = subprocess.Popen("nvidia-smi -L", universal_newlines=True, shell=True, stdout=subprocess.PIPE)
    for line in p.stdout:
        if re.search(r"MIG\s+\dg\.\d+gb", line):
            return True
    return False


def get_gpu_uuid_from_mig_uuid(mig_uuid):
    """Return GPU UUID corresponding to MIG UUID.
    """
    gpu_mig_slice_mapping = MIGConfiguration.get_gpu_mig_slice_mapping()
    ret_gpu_uuid = ""
    for gpu_uuid, mig_slices in gpu_mig_slice_mapping.items():
        mig_uuids = [mig_slice.uuid for mig_slice in mig_slices]
        if mig_uuid in mig_uuids:
            ret_gpu_uuid = gpu_uuid
            break
    return ret_gpu_uuid


def get_system():
    """Return a System object that describes computer system.
    """
    # Quick path for CPU machines
    if os.environ.get("USE_CPU") == "1":
        cpu_info = run_command("lscpu | grep name", get_output=True, tee=False)
        model_name = cpu_info[0].replace("Model name:", "").strip()
        if "6258R" in model_name:
            return KnownSystems.Triton_CPU_2S_6258R.get_match("2S_6258R", 1)
        elif "8380H" in model_name:
            return KnownSystems.Triton_CPU_4S_8380H.get_match("4S_8380H", 1)
        else:
            raise RuntimeError("Cannot find valid configs for {:}.".format(model_name))

    # Check if system is Xavier
    if is_xavier():
        # Jetson Xavier is the only supported aarch64 platform.
        with open("/sys/firmware/devicetree/base/model") as product_f:
            product_name = product_f.read()
        if "jetson" in product_name.lower():
            if "AGX" in product_name:
                return KnownSystems.AGX_Xavier.get_match("Jetson-AGX", 1)
            elif "NX" in product_name:
                return KnownSystems.Xavier_NX.get_match("Xavier NX", 1)
            else:
                raise RuntimeError("Unrecognized aarch64 device. Only AGX Xavier and Xavier NX are supported.")

    # Check if MIG is enabled
    mig_conf = None
    if check_mig_enabled():
        mig_conf = MIGConfiguration.from_nvidia_smi()
        if mig_conf.num_mig_slices() == 0:
            logging.warn("MIG is enabled, but no instances were detected.")
        else:
            logging.info("Found {:} MIG compute instances".format(mig_conf.num_mig_slices()))

    # TODO: Investigate using py-nvml to get this information, instead of nvidia-smi. It may break on aarch64.
    # Get GPU name and count from nvidia-smi
    nvidia_smi_out = run_command("CUDA_VISIBLE_ORDER=PCI_BUS_ID nvidia-smi --query-gpu=gpu_name,pci.device_id,uuid --format=csv", get_output=True, tee=False)

    # Remove first line (CSV column names) and strip empty lines
    tmp = [line for line in nvidia_smi_out[1:] if len(line) > 0]
    uuid2index = {line.split(',')[2].strip(): i for i, line in enumerate(tmp)}

    # If CUDA_VISIBLE_DEVICES is set, apply it manually, as nvidia-smi doesn't obey it.
    # Indexing is correct, as we set CUDA_VISIBLE_ORDER to PCI_BUS_ID.
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        seen_uuids = set()
        indices = []
        for g in os.environ.get("CUDA_VISIBLE_DEVICES").split(","):
            if g.isnumeric():
                indices.append(int(g))
            else:
                uuid = ""
                if g.startswith("GPU-"):
                    uuid = g
                elif g.startswith("MIG-"):
                    uuid = get_gpu_uuid_from_mig_uuid(g)
                else:
                    raise RuntimeError("Invalid CUDA_VISIBILE_DEVICES")
                if uuid not in seen_uuids:
                    seen_uuids.add(uuid)
                    indices.append(uuid2index[uuid])

        tmp = [tmp[i] for i in indices]

    count_actual = len(tmp)
    if count_actual == 0:
        raise RuntimeError("nvidia-smi did not detect any GPUs:\n{:}".format(nvidia_smi_out))

    name, pci_id, uuid = tmp[0].split(", ")
    assert(pci_id[-4:] == "10DE")  # 10DE is NVIDIA PCI vendor ID
    pci_id = pci_id.split("x")[1][:4]  # Get the relevant 4 digit hex

    system = None
    for sysclass in KnownSystems.get_all_system_classes():
        system = sysclass.get_match(name, count_actual, pci_id=pci_id, mig_conf=mig_conf)
        if system:
            break

    if system is None:
        raise RuntimeError("Cannot find valid configs for {:d}x {:}. Please follow performance_tuning_guide.md to add support for a new GPU.".format(count_actual, name))

    return system


def run_command(cmd, get_output=False, tee=True, custom_env=None):
    """
    Runs a command.

    Args:
        cmd (str): The command to run.
        get_output (bool): If true, run_command will return the stdout output. Default: False.
        tee (bool): If true, captures output (if get_output is true) as well as prints output to stdout. Otherwise, does
            not print to stdout.
    """
    logging.info("Running command: {:}".format(cmd))
    if not get_output:
        return subprocess.check_call(cmd, shell=True)
    else:
        output = []
        if custom_env is not None:
            logging.info("Overriding Environment")
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True, env=custom_env)
        else:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        for line in iter(p.stdout.readline, b""):
            line = line.decode("utf-8")
            if tee:
                sys.stdout.write(line)
                sys.stdout.flush()
            output.append(line.rstrip("\n"))
        ret = p.wait()
        if ret == 0:
            return output
        else:
            raise subprocess.CalledProcessError(ret, cmd)


def args_to_string(d, blacklist=[], delimit=True, double_delimit=False):
    flags = []
    for flag in d:
        # Skip unset
        if d[flag] is None:
            continue
        # Skip blacklisted
        if flag in blacklist:
            continue
        if type(d[flag]) is bool:
            if d[flag] is True:
                flags.append("--{:}=true".format(flag))
            elif d[flag] is False:
                flags.append("--{:}=false".format(flag))
        elif type(d[flag]) in [int, float] or not delimit:
            flags.append("--{:}={:}".format(flag, d[flag]))
        else:
            if double_delimit:
                flags.append("--{:}=\\\"{:}\\\"".format(flag, d[flag]))
            else:
                flags.append("--{:}=\"{:}\"".format(flag, d[flag]))
    return " ".join(flags)


def flags_bool_to_int(d):
    for flag in d:
        if type(d[flag]) is bool:
            if d[flag]:
                d[flag] = 1
            else:
                d[flag] = 0
    return d


def dict_get(d, key, default=None):
    """Return non-None value for key from dict. Use default if necessary."""

    val = d.get(key, default)
    return default if val is None else val
