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
from code.common import logging, dict_get
from code.common.constants import Benchmark, Scenario
import shutil
import textwrap
import json

TENSORRT_VERSION = ""
TRITON_VERSION = ""

# option name to config file map
options_map = {
    "single_stream_expected_latency_ns": "target_latency",
    "single_stream_target_latency_percentile": "target_latency_percentile",
    "offline_expected_qps": "target_qps",
    "server_target_qps": "target_qps",
    "server_target_latency_percentile": "target_latency_percentile",
    "server_target_latency_ns": "target_latency",
}

parameter_scaling_map = {
    "target_latency": 1 / 1000000.0,
    "target_latency_percentile": 100.0,
}


def generate_measurements_entry(system_name, short_benchmark_name, full_benchmark_name, scenario, input_dtype, precision, flag_dict):
    measurements_dir = "measurements/{:}/{:}/{:}".format(system_name, full_benchmark_name, scenario.valstr())
    os.makedirs(measurements_dir, exist_ok=True)

    # Override perf_sample_count. Make sure it's larger than the values required by the rules.
    flag_dict["performance_sample_count_override"] = flag_dict["performance_sample_count"]

    # Copy mlperf.conf
    mlperf_conf_path = os.path.join(measurements_dir, "mlperf.conf")
    if "mlperf_conf_path" not in flag_dict:
        flag_dict["mlperf_conf_path"] = mlperf_conf_path
    generate_mlperf_conf(mlperf_conf_path)

    # Auto-generate user.conf
    user_conf_path = os.path.join(measurements_dir, "user.conf")
    if "user_conf_path" not in flag_dict:
        flag_dict["user_conf_path"] = user_conf_path
    generate_user_conf(user_conf_path, scenario, flag_dict)

    # Write out README.md (required file by MLPerf Submission rules)
    readme_path = os.path.join(measurements_dir, "README.md")
    generate_readme(readme_path, system_name, short_benchmark_name, scenario)

    # Generate calibration_process.adoc
    calibration_process_path = os.path.join(measurements_dir, "calibration_process.adoc")
    generate_calibration_process(system_name, calibration_process_path, short_benchmark_name, scenario)

    # Generate system json
    system_json_path = os.path.join(measurements_dir, "{:}_{:}.json".format(system_name, scenario.valstr()))
    generate_system_json(system_name, system_json_path, short_benchmark_name, input_dtype, precision)

    # Write powersetting.adoc for MaxQ systems:
    if system_name.endswith("MaxQ"):
        powersetting_path = os.path.join("measurements", system_name, full_benchmark_name, "powersetting.adoc")
        generate_powersetting_adoc(powersetting_path)


def generate_mlperf_conf(mlperf_conf_path):
    shutil.copyfile("build/inference/mlperf.conf", mlperf_conf_path)


def generate_user_conf(user_conf_path, scenario, flag_dict):
    # Required settings for each scenario
    common_required = ["performance_sample_count_override"]
    required_settings_map = {
        Scenario.SingleStream: [] + common_required,  # "single_stream_expected_latency_ns", See: https://github.com/mlcommons/inference/issues/471
        Scenario.Offline: ["offline_expected_qps"] + common_required,
        Scenario.Server: ["server_target_qps"] + common_required,
    }

    # Optional settings that we support overriding
    common_optional = ["min_query_count", "max_query_count", "min_duration", "max_duration"]
    optional_settings_map = {
        Scenario.SingleStream: ["single_stream_target_latency_percentile"] + common_optional,
        Scenario.Offline: [] + common_optional,
        Scenario.Server: ["server_target_latency_percentile", "server_target_latency_ns"] + common_optional,
    }

    with open(user_conf_path, 'w') as f:
        for param in required_settings_map[scenario]:
            param_name = param
            if param in options_map:
                param_name = options_map[param]
            value = flag_dict[param]
            if param_name in parameter_scaling_map:
                value = value * parameter_scaling_map[param_name]
            f.write("*.{:}.{:} = {:}\n".format(scenario.valstr(), param_name, value))
            flag_dict[param] = None

        for param in optional_settings_map[scenario]:
            if param not in flag_dict:
                continue
            param_name = param
            if param in options_map:
                param_name = options_map[param]
            value = flag_dict[param]
            if param_name in parameter_scaling_map:
                value = value * parameter_scaling_map[param_name]
            f.write("*.{:}.{:} = {:}\n".format(scenario.valstr(), param_name, value))
            flag_dict[param] = None


def generate_readme(readme_path, system_name, short_benchmark_name, scenario):

    if "Triton_CPU" in system_name:
        readme_str = textwrap.dedent("""\
        To run this benchmark, first follow the setup steps in `closed/NVIDIA/README_Triton_CPU.md`. Then to run the harness:

        ```
        make run_harness RUN_ARGS="--benchmarks={benchmark} --scenarios={scenario} --test_mode=AccuracyOnly"
        make run_harness RUN_ARGS="--benchmarks={benchmark} --scenarios={scenario} --test_mode=PerformanceOnly"
        ```

        For more details, please refer to `closed/NVIDIA/README_Triton_CPU.md`.""".format(
            benchmark=short_benchmark_name,
            scenario=scenario.valstr()))
    else:
        readme_str = textwrap.dedent("""\
        To run this benchmark, first follow the setup steps in `closed/NVIDIA/README.md`. Then to generate the TensorRT engines and run the harness:

        ```
        make generate_engines RUN_ARGS="--benchmarks={benchmark} --scenarios={scenario}"
        make run_harness RUN_ARGS="--benchmarks={benchmark} --scenarios={scenario} --test_mode=AccuracyOnly"
        make run_harness RUN_ARGS="--benchmarks={benchmark} --scenarios={scenario} --test_mode=PerformanceOnly"
        ```

        For more details, please refer to `closed/NVIDIA/README.md`.""".format(
            benchmark=short_benchmark_name,
            scenario=scenario.valstr()))

    if "HeteroMultiUse" in system_name:
        readme_str = textwrap.dedent("""\
        To run this benchmark, first follow the setup steps in `closed/NVIDIA/README.md`. Then to generate the TensorRT
        engines and run the harness, first read the **Using Multiple MIG slices** section in `closed/NVIDIA/README.md`.
        Then follow the instructions in `closed/NVIDIA/documentation/heterogeneous_mig.md` to run benchmarks.""")

    with open(readme_path, 'w') as f:
        f.write(readme_str)


def generate_calibration_process(system_name, calibration_process_path, short_benchmark_name, scenario):
    if "Triton_CPU" in system_name:
        calibration_process_str = textwrap.dedent("""\
        To calibrate this benchmark, please follow the steps in `closed/NVIDIA/calibration_triton_cpu/OpenVINO/{benchmark}/README.md`.""".format(
            benchmark=short_benchmark_name))
    else:
        calibration_process_str = textwrap.dedent("""\
        To calibrate this benchmark, first follow the setup steps in `closed/NVIDIA/README.md`.

        ```
        make calibrate RUN_ARGS="--benchmarks={benchmark} --scenarios={scenario}"
        ```

        For more details, please refer to `closed/NVIDIA/README.md`.""".format(
            benchmark=short_benchmark_name,
            scenario=scenario.valstr()))
    with open(calibration_process_path, 'w') as f:
        f.write(calibration_process_str)


def generate_system_json(system_name, system_json_path, short_benchmark_name, input_dtype, precision):

    if "Triton_CPU" in system_name:
        starting_weights_filename_map = {
            Benchmark.ResNet50: "The original weight filename: https://zenodo.org/record/2535873/files/resnet50_v1.pb",
            Benchmark.SSDResNet34: "The original weight filename: https://zenodo.org/record/3228411/files/resnet34-ssd1200.onnx",
            Benchmark.BERT: "The original weight filename: bert_large_v1_1_fake_quant.onnx",
            Benchmark.UNET3D: "The original weight filename: https://zenodo.org/record/3928973/files/224_224_160.onnx",
        }
        weight_transformations_map = {
            Benchmark.ResNet50: "We transform the original fp32 weight to int8 weight using symmetric quantization.",
            Benchmark.SSDResNet34: "We transfer the weight from fp32 datatype in ONNX file to int8 datatype in OpenVino IR file.",
            Benchmark.BERT: "We transfer the weight from int8 datatype in ONNX file to int8 datatype in OpenVino IR file.",
            Benchmark.UNET3D: "We transfer the weight from fp32 datatype in ONNX file to int8 datatype in OpenVino IR file.",
        }
        if Benchmark.BERT == short_benchmark_name:
            precision = "int8"
    else:
        starting_weights_filename_map = {
            Benchmark.ResNet50: "resnet50_v1.onnx",
            Benchmark.SSDResNet34: "resnet34-ssd1200.pytorch",
            Benchmark.SSDMobileNet: "ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb",
            Benchmark.RNNT: "DistributedDataParallel_1576581068.9962234-epoch-100.pt",
            Benchmark.DLRM: "tb00_40M.pt",
            Benchmark.BERT: "bert_large_v1_1_fake_quant.onnx",
            Benchmark.UNET3D: "224_224_160_dyanmic_bs.onnx",
        }
        weight_transformations_map = {
            Benchmark.ResNet50: "quantization, affine fusion",
            Benchmark.SSDResNet34: "quantization, affine fusion",
            Benchmark.SSDMobileNet: "quantization, affine fusion",
            Benchmark.RNNT: "quantization, affine fusion",
            Benchmark.DLRM: "quantization, affine fusion",
            Benchmark.BERT: "quantization, affine fusion",
            Benchmark.UNET3D: "quantization, affine fusion",
        }

    data = {
        "input_data_types": input_dtype,
        "retraining": "No",
        "starting_weights_filename": starting_weights_filename_map[short_benchmark_name],
        "weight_data_types": precision,
        "weight_transformations": weight_transformations_map[short_benchmark_name]
    }

    with open(system_json_path, 'w') as f:
        json.dump(data, f, indent=4, sort_keys=True)


def generate_powersetting_adoc(powersetting_path):
    powersetting_str = textwrap.dedent("""\
    ## An example of an unstructured document for Power management settings to reproduce Perf, Power results

    # Boot/BIOS Firmware Settings

    None

    # Management Firmware Settings

    None

    # Power Management Settings  (command line or other)

    Run the scripts described in our code. See the **How to collect power measurements while running the harness**
    section of the README.md located in closed/NVIDIA.
    """)
    with open(powersetting_path, 'w') as f:
        f.write(powersetting_str)
