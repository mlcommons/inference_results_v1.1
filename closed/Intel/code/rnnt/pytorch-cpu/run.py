# Copyright 2020 The MLPerf Authors. All Rights Reserved.
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
# =============================================================================

import argparse
import mlperf_loadgen as lg
import subprocess

import os
from pathlib import Path

MLPERF_CONF = Path(os.path.dirname(os.path.realpath(__file__))) / "./mlperf.conf"
MLPERF_CONF = MLPERF_CONF.resolve()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["pytorch"], default="pytorch", help="Backend")
    parser.add_argument("--scenario", choices=["SingleStream", "Offline", "Server"], default="Offline", help="Scenario")
    parser.add_argument("--accuracy", action="store_true", help="enable accuracy pass")
    parser.add_argument("--mlperf_conf", default=str(MLPERF_CONF), help="mlperf rules config")
    parser.add_argument("--user_conf", default="user.conf", help="user config for user LoadGen settings such as target QPS")
    parser.add_argument("--pytorch_config_toml", default="pytorch/configs/rnnt.toml")
    parser.add_argument("--pytorch_checkpoint", required=True)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--perf_count", type=int, default=None)
    parser.add_argument("--profile", choices=["True", "Split", "False"], default="False")
    parser.add_argument("--verbose", choices=["0", "1", "2"], default="0")
    parser.add_argument("--bf16", type=bool, default=False)
    parser.add_argument("--int8", type=bool, default=False)
    parser.add_argument("--warmup", action="store_true", help="enable warmup")
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--configure_path", default="")
    parser.add_argument("--debug", action='store_true', default=False)
    parser.add_argument("--machine_conf", default="machine.conf", help="config for hetero instances. Format: run_mode/core_count/num_instances/max_seq_len/batch_size")
    args = parser.parse_args()
    return args


scenario_map = {
    "SingleStream": lg.TestScenario.SingleStream,
    "Offline": lg.TestScenario.Offline,
    "Server": lg.TestScenario.Server,
}


def main():
    args = get_args()

    settings = lg.TestSettings()
    settings.scenario = scenario_map[args.scenario]
    settings.FromConfig(args.mlperf_conf, "rnnt", args.scenario)
    settings.FromConfig(args.user_conf, "rnnt", args.scenario)

    if args.accuracy:
        settings.mode = lg.TestMode.AccuracyOnly
    else:
        settings.mode = lg.TestMode.PerformanceOnly

    log_path = args.log_dir
    os.makedirs(log_path, exist_ok=True)
    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = log_path
    log_output_settings.copy_summary_to_stdout = True
    log_settings = lg.LogSettings()
    log_settings.log_output = log_output_settings

    print ("Checking args: int8={}, bf16={}".format(args.int8, args.bf16))
    if args.backend == "pytorch":
        from pytorch_SUT import PytorchSUT
        sut = PytorchSUT(args.pytorch_config_toml, args.pytorch_checkpoint,
                         args.dataset_dir, args.manifest, args.perf_count,
                         args.machine_conf, args.debug, args.profile, args.verbose,
                         args.bf16, args.warmup, args.int8, args.configure_path)
    else:
        raise ValueError("Unknown backend: {:}".format(args.backend))

    print("Running Loadgen test...")
    lg.StartTestWithLogSettings(sut.sut, sut.qsl.qsl, settings, log_settings)

    if args.accuracy:
        cmd = f"python3 accuracy_eval.py --log_dir {log_path} --dataset_dir {args.dataset_dir} --manifest {args.manifest}"
        print(f"Running accuracy script: {cmd}")
        subprocess.check_call(cmd, shell=True)

    lg.DestroySUT(sut.sut)

    print("Done!")



if __name__ == "__main__":
    main()
