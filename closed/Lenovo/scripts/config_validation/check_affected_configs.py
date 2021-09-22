#!/usr/bin/env python3
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

import argparse
import copy
import json
from enum import Enum, unique
from importlib import import_module
from typing import Any, Dict, Tuple
from code.common.constants import Benchmark, Scenario
from configs.configuration import ConfigRegistry

__doc__ = """
This script will output a list of changes to a given config, and assumes the original config is stored at
config.<benchmark>.<scenario>.old (where old is a Python module).

Users will be shown a set of changes to configs by system_id, and be prompted to confirm that those changes are
expected. If the user enters a negative input (i.e. 'no'), the script will exit with non-zero exit code.
"""

# This script is run from a git hook, which is not supposed to be an interactive setting.
# Force stdin to take user input through command line input
if __name__ == "__main__":
    sys.stdin = open("/dev/tty", "r")


@unique
class Change(Enum):
    Addition = "addition"
    Removal = "removal"
    Edit = "edit"


def find_changes(d_before, d_after):
    """
    Returns a dictionary of changes in the format:
        {
            <system id>: {
                <changed key>: <Change type>,
                ...
            },
            ...
        }
    The changes should describe the differences between d_before and d_after.
    """
    changes = dict()
    for k in d_after:
        if k not in d_before:
            changes[k] = Change.Addition
        elif type(d_before[k]) is dict and type(d_after[k]) is dict:
            nested = find_changes(d_before[k], d_after[k])
            if len(nested) > 0:
                changes[k] = nested
        elif d_before[k] != d_after[k]:
            changes[k] = Change.Edit

    # Apply removals
    for k in d_before:
        if k not in d_after:
            changes[k] = Change.Removal

    return changes


def extract_changes(d):
    """
    Returns a sub-dict of d that only contains k-v pairs where v is of type 'Change', not to be confused with
    'find_changes'.
    """
    changes = dict()
    for k in d:
        if type(d[k]) is Change:
            changes[k] = d[k]
        elif type(d[k]) is dict:
            nested = extract_changes(d[k])
            if len(nested) > 0:
                changes[k] = nested
    return changes


def prompt_user(msg):
    while True:
        user_in = input(msg)
        if user_in.lower().startswith("y"):
            return True
        elif user_in.lower().startswith("n"):
            return False


def update_nested(d: Dict[str, Any], v: Any):
    """
    Updates a dictionary such that every non-dict value is set to v. If a value is of type dict, this method will be
    called recursively on that dictionary.

    Args:
        d (Dict[str, Any]): The dictionary to update
        v: (Any): The value to set each key to in d
    """
    for k in d:
        if isinstance(d[k], dict):
            update_nested(d[k], v)
        else:
            d[k] = v


def load_old_configs(benchmark: Benchmark, scenario: Scenario):
    """
    Load the module from configs.benchmark.scenario.old.

    Args:
        benchmark (Benchmark):
            The benchmark of the config with diffs
        scenario (Scenario):
            The scenario of the config with diffs
    """
    try:
        import_module(f"configs.old_conf_to_check")
        return True
    except ModuleNotFoundError:
        return False


def get_raw_configs_and_traces(benchmark: Benchmark, scenario: Scenario) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Returns the raw configs and traces of a benchmark-scenario pair from the config registry. The ConfigRegistry should
    not have any other benchmark-scenario pairs loaded, and this is asserted at the beginning of this function.

    Args:
        benchmark (Benchmark):
            The benchmark of the config
        scenario (Scenario):
            The scenario of the config

    Returns:
        dict: A dictionary of {system_id -> {workload_setting -> config.as_dict() output}}
        dict: A dictionary of {system_id -> {workload_setting -> config.get_field_trace() output}}
    """
    registry = ConfigRegistry._registry.tree
    assert list(registry.keys()) == [benchmark]
    assert list(registry[benchmark].keys()) == [scenario]

    configs = dict()
    traces = dict()
    registry = registry[benchmark][scenario]
    for system_id in registry:
        configs[system_id] = dict()
        traces[system_id] = dict()
        for workload_setting in registry[system_id]:
            configs[system_id][workload_setting] = registry[system_id][workload_setting].as_dict()
            traces[system_id][workload_setting] = registry[system_id][workload_setting].get_field_trace().trace
    return configs, traces


def user_verify_changes(benchmark: Benchmark, scenario: Scenario):
    """
    Asks users to verify the changes to the file.

    Args:
        benchmark (Benchmark):
            The benchmark of the config with diffs
        scenario (Scenario):
            The scenario of the config with diffs
    """
    load_old_configs(benchmark, scenario)
    old_config, old_trace = get_raw_configs_and_traces(benchmark, scenario)
    ConfigRegistry._reset()

    ConfigRegistry.load_configs(benchmark, scenario)
    new_config, new_trace = get_raw_configs_and_traces(benchmark, scenario)
    ConfigRegistry._reset()

    changes = find_changes(old_config, new_config)
    extracted_changes = extract_changes(changes)

    for system_id in extracted_changes:
        system_id_edit_state = Change.Edit
        if extracted_changes[system_id] == Change.Addition:
            extracted_changes[system_id] = copy.deepcopy(new_config[system_id])
            update_nested(extracted_changes[system_id], Change.Addition)
            system_id_edit_state = Change.Addition
        elif extracted_changes[system_id] == Change.Removal:
            extracted_changes[system_id] = copy.deepcopy(old_config[system_id])
            update_nested(extracted_changes[system_id], Change.Removal)
            system_id_edit_state = Change.Removal

        for workload_setting in extracted_changes[system_id]:
            print(f"{system_id} with {workload_setting}")
            if system_id_edit_state == Change.Addition:
                print(f"    !! This is a NEW system_id for {str(benchmark.value)} {str(scenario.value)}")
            elif system_id_edit_state == Change.Removal:
                print(f"    !! This system_id was REMOVED for {str(benchmark.value)} {str(scenario.value)}")

            workload_setting_edit_state = Change.Edit
            if extracted_changes[system_id][workload_setting] == Change.Addition:
                extracted_changes[system_id][workload_setting] = copy.deepcopy(new_config[system_id][workload_setting])
                update_nested(extracted_changes[system_id][workload_setting], Change.Addition)
                workload_setting_edit_state = Change.Addition
            elif extracted_changes[system_id][workload_setting] == Change.Removal:
                extracted_changes[system_id][workload_setting] = copy.deepcopy(old_config[system_id][workload_setting])
                update_nested(extracted_changes[system_id][workload_setting], Change.Removal)
                workload_setting_edit_state = Change.Removal
            if workload_setting_edit_state == Change.Addition:
                print(f"    !! This is a NEW workload setting for {str(benchmark.value)} {str(scenario.value)}")
            elif workload_setting_edit_state == Change.Removal:
                print(f"    !! This workload setting was REMOVED for {str(benchmark.value)} {str(scenario.value)}")

            for k in extracted_changes[system_id][workload_setting]:
                indent_str = " " * 4
                change = extracted_changes[system_id][workload_setting][k]

                if change == Change.Addition:
                    new_val = new_config[system_id][workload_setting].get(k, Change.Edit)
                    trace = new_trace[system_id][workload_setting][k][0]
                    trace_str = f"{trace.klass}:L{trace.lineno}"
                    print(f"{indent_str}(NEW) '{k}': {new_val} -- {trace_str}")
                elif change == Change.Removal:
                    old_val = old_config[system_id][workload_setting].get(k, Change.Edit)
                    trace = old_trace[system_id][workload_setting][k][0]
                    trace_str = f"{trace.klass}:L{trace.lineno}"
                    print(f"{indent_str}(REMOVED) '{k}': {old_val} -- {trace_str}")
                else:
                    old_val = old_config[system_id][workload_setting].get(k, Change.Edit)
                    new_val = new_config[system_id][workload_setting].get(k, Change.Edit)
                    # k must either be an edit or a dict. Either way, print it as a bulk change.
                    trace = new_trace[system_id][workload_setting][k][0]
                    trace_str = f"{trace.klass}:L{trace.lineno}"
                    print(f"{indent_str}'{k}': {old_val} -> {new_val} -- {trace_str}")

            if not prompt_user("Confirm that these changes are expected [y/n]: "):
                raise RuntimeError("Aborted by user.")
            print("\n============================\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "benchmark",
        help="Benchmark to check for diffs",
        choices=Benchmark.as_strings(),
    )
    parser.add_argument(
        "scenario",
        help="Scenario to check for diffs",
        choices=Scenario.as_strings(),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    user_verify_changes(Benchmark.get_match(args.benchmark), Scenario.get_match(args.scenario))


if __name__ == "__main__":
    main()
