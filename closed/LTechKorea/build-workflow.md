# MLPerf Inference Build Workflow

by jay.huh@ltechkorea.com

> The MLPerf inference v1.0 suite contains the following models for benchmark:

- Resnet50
- SSD-Resnet34
- BERT
- DLRM
- RNN-T
- 3D U-Net

## Build Docker Images

```bash
make prebuild
```

- **(DOCKER)** Launch Docker
  ```bash
  make launch_docker
  ```

## Build Pre-requisites : `triton`, `plugin(power-dev)`, `loadgen`, and `harness`

  ```bash
  make build
  or
  make launch_docker DOCKER_COMMAND='build'
  ```

## Download All Dataset( check ) and Al Model

  ```bash
  make download_data [ BENCHMARKS=... ]
  make download_model [ BENCHMARKS=... ]
  ```

## **(DOCKER)** Preprocessing All

  ```bash
  make preprocess_data [ BENCHMARKS=... ]
  or
  make launch_docker DOCKER_COMMAND='make preprocess_data [BENCHMARKS=...]'
  ```

## **(DOCKER)** Run Benchmark

```bash
# run the performance benchmark
make launch_docker DOCKER_COMMAND='make run RUN_ARGS="--benchmarks=[benchmark] --scenarios=Server --config_ver=default --test_mode=PerformanceOnly"'

# run the accuracy benchmark
make launch_docker DOCKER_COMMAND='make run RUN_ARGS="--benchmarks=[benchmark] --scenarios=Server --config_ver=default --test_mode=AccuracyOnly"'
```
Note: `make run` rule includes `generate_engine` and `run_harness`.

- Individual Run

```bash
# generates the TRT engines with the specified config. In this case it generates engine for both Offline and Server scenario
make generate_engines RUN_ARGS="--benchmarks=[benchmark] --scenarios=Offline,Server --config_ver=default"
or
make launch_docker DOCKER_COMMAND='make generate_engines RUN_ARGS="--benchmarks=[benchmark] --scenarios=Offline,Server --config_ver=default"'

# run the performance benchmark
make run_harness RUN_ARGS="--benchmarks=[benchmark] --scenarios=Offline --config_ver=default --test_mode=PerformanceOnly"
make run_harness RUN_ARGS="--benchmarks=[benchmark] --scenarios=Server --config_ver=default --test_mode=PerformanceOnly"
or
make launch_docker DOCKER_COMMAND='make run_harness RUN_ARGS="--benchmarks=[benchmark] --scenarios=Server --config_ver=default --test_mode=PerformanceOnly"'

# run the accuracy benchmark
make run_harness RUN_ARGS="--benchmarks=[benchmark] --scenarios=Offline --config_ver=default --test_mode=AccuracyOnly"
make run_harness RUN_ARGS="--benchmarks=[benchmark] --scenarios=Server --config_ver=default --test_mode=AccuracyOnly"
or
make launch_docker DOCKER_COMMAND='make run_harness RUN_ARGS="--benchmarks=[benchmark] --scenarios=Server --config_ver=default --test_mode=AccuracyOnly"'
```

## Compliance

```bash
make launch_docker DOCKER_COMMAND='make run_audit_harness'
```

## Result Logs and Compliance Logs

```bash
make make update_results'
make make update_compliance'
```

---

## Resnet50

### Download Dataset and Model

  ```bash
  make download_model BENCHMARK=resnet50
  make download_data BENCHMARK=resnet50
  ```

### **(DOCKER)** Preprocessing

### **(DOCKER)** Generate the TensorRT engines:

```bash
# generates the TRT engines with the specified config. In this case it generates engine for both Offline and Server scenario
make generate_engines RUN_ARGS="--benchmarks=resnet50 --scenarios=Offline,Server --config_ver=default"
or
make launch_docker DOCKER_COMMAND='make generate_engines RUN_ARGS="--benchmarks=resnet50 --scenarios=Offline,Server --config_ver=default"'
```

### **(DOCKER)** Run the benchmark:

```bash
# run the performance benchmark
make run_harness RUN_ARGS="--benchmarks=resnet50 --scenarios=Offline --config_ver=default --test_mode=PerformanceOnly"
make run_harness RUN_ARGS="--benchmarks=resnet50 --scenarios=Server --config_ver=default --test_mode=PerformanceOnly"

# run the accuracy benchmark
make run_harness RUN_ARGS="--benchmarks=resnet50 --scenarios=Offline --config_ver=default --test_mode=AccuracyOnly"
make run_harness RUN_ARGS="--benchmarks=resnet50 --scenarios=Server --config_ver=default --test_mode=AccuracyOnly"
```

## SSD-Resnet34

### Download Dataset and Model

```bash
make download_model BENCHMARKS=ssd-resnet34
make download_data BENCHMARKS=ssd-resnet34
make preprocess_data BENCHMARKS=ssd-resnet34
```

### **(DOCKER)** Preprocessing

### **(DOCKER)** Generate the TensorRT engines:

```bash
# generates the TRT engines with the specified config. In this case it generates engine for both Offline and Server scenario
make generate_engines RUN_ARGS="--benchmarks=ssd-resnet34 --scenarios=Offline,Server --config_ver=default"
```

### **(DOCKER)** Run the benchmark:

```bash
# run the performance benchmark
make run_harness RUN_ARGS="--benchmarks=ssd-resnet34 --scenarios=Offline --config_ver=default --test_mode=PerformanceOnly"
make run_harness RUN_ARGS="--benchmarks=ssd-resnet34 --scenarios=Server --config_ver=default --test_mode=PerformanceOnly"
 
# run the accuracy benchmark
make run_harness RUN_ARGS="--benchmarks=ssd-resnet34 --scenarios=Offline --config_ver=default --test_mode=AccuracyOnly"
make run_harness RUN_ARGS="--benchmarks=ssd-resnet34 --scenarios=Server --config_ver=default --test_mode=AccuracyOnly"
```

### Run the RNN-T benchmark

```bash
make download_model BENCHMARKS=rnnt
make download_data BENCHMARKS=rnnt
make preprocess_data BENCHMARKS=rnnt
```

### **(DOCKER)** Preprocessing

### **(DOCKER)** Generate the TensorRT engines:

```bash
# generates the TRT engines with the specified config. In this case it generates engine for both Offline and Server scenario
make generate_engines RUN_ARGS="--benchmarks=rnnt --scenarios=Offline,Server --config_ver=default"
```

### **(DOCKER)** Run the benchmark:

```bash
# run the performance benchmark
make run_harness RUN_ARGS="--benchmarks=rnnt --scenarios=Offline --config_ver=default --test_mode=PerformanceOnly"
make run_harness RUN_ARGS="--benchmarks=rnnt --scenarios=Server --config_ver=default --test_mode=PerformanceOnly"

# run the accuracy benchmark
make run_harness RUN_ARGS="--benchmarks=rnnt --scenarios=Offline --config_ver=default --test_mode=AccuracyOnly"
make run_harness RUN_ARGS="--benchmarks=rnnt --scenarios=Server --config_ver=default --test_mode=AccuracyOnly"
```

> The BERT, DLRM, and 3D U-Net benchmarks have high accuracy targets.

## Run the BERT benchmark

```bash
make download_model BENCHMARKS=bert
make download_data BENCHMARKS=bert
make preprocess_data BENCHMARKS=bert
```

### **(DOCKER)** Preprocessing

### **(DOCKER)** Generate the TensorRT engines:

```bash
# generates the TRT engines with the specified config. In this case it generates engine for both Offline and Server scenario and also for default and high accuracy targets.
make generate_engines RUN_ARGS="--benchmarks=bert --scenarios=Offline,Server --config_ver=default,high_accuracy"
```

### **(DOCKER)** Run the benchmark:

```bash
# run the performance benchmark
make run_harness RUN_ARGS="--benchmarks=bert --scenarios=Offline --config_ver=default --test_mode=PerformanceOnly"
make run_harness RUN_ARGS="--benchmarks=bert --scenarios=Server --config_ver=default --test_mode=PerformanceOnly"

make run_harness RUN_ARGS="--benchmarks=bert --scenarios=Offline --config_ver=high_accuracy --test_mode=PerformanceOnly"
make run_harness RUN_ARGS="--benchmarks=bert --scenarios=Server --config_ver=high_accuracy --test_mode=PerformanceOnly"
 
# run the accuracy benchmark
make run_harness RUN_ARGS="--benchmarks=bert --scenarios=Offline --config_ver=default --test_mode=AccuracyOnly"
make run_harness RUN_ARGS="--benchmarks=bert --scenarios=Server --config_ver=default --test_mode=AccuracyOnly"

make run_harness RUN_ARGS="--benchmarks=bert --scenarios=Offline --config_ver=high_accuracy --test_mode=AccuracyOnly"
make run_harness RUN_ARGS="--benchmarks=bert --scenarios=Server --config_ver=high_accuracy --test_mode=AccuracyOnly"
```

## Run the DLRM benchmark

To set up the DLRM dataset and model to run the inference:

If you already downloaded and preprocessed the datasets, go to step 5.
Download the Criteo Terabyte dataset.
Extract the images to $MLPERF_SCRATCH_PATH/data/criteo/ directory.
Run the following commands:

```bash
make download_model BENCHMARKS=dlrm
make preprocess_data BENCHMARKS=dlrm
```

### **(DOCKER)** Preprocessing

### **(DOCKER)** Generate the TensorRT engines:

```bash
# generates the TRT engines with the specified config. In this case it generates engine for both Offline and Server scenario and also for default and high accuracy targets.
make generate_engines RUN_ARGS="--benchmarks=dlrm --scenarios=Offline,Server --config_ver=default, high_accuracy"
```

### **(DOCKER)** Run the benchmark:

```bash
# run the performance benchmark
make run_harness RUN_ARGS="--benchmarks=dlrm --scenarios=Offline --config_ver=default --test_mode=PerformanceOnly"
make run_harness RUN_ARGS="--benchmarks=dlrm --scenarios=Server --config_ver=default --test_mode=PerformanceOnly"

make run_harness RUN_ARGS="--benchmarks=dlrm --scenarios=Offline --config_ver=high_accuracy --test_mode=PerformanceOnly"
make run_harness RUN_ARGS="--benchmarks=dlrm --scenarios=Server --config_ver=high_accuracy --test_mode=PerformanceOnly"
 
# run the accuracy benchmark
make run_harness RUN_ARGS="--benchmarks=dlrm --scenarios=Offline --config_ver=default --test_mode=AccuracyOnly"
make run_harness RUN_ARGS="--benchmarks=dlrm --scenarios=Server --config_ver=default --test_mode=AccuracyOnly"

make run_harness RUN_ARGS="--benchmarks=dlrm --scenarios=Offline --config_ver=high_accuracy --test_mode=AccuracyOnly"
make run_harness RUN_ARGS="--benchmarks=dlrm --scenarios=Server --config_ver=high_accuracy --test_mode=AccuracyOnly"
```

## Run the 3D U-Net benchmark

Note: This benchmark only has the Offline scenario.

To set up the 3D U-Net dataset and model to run the inference:

If you already downloaded and preprocessed the datasets, go to step 5.
Download the BraTS challenge data.
Extract the images to the $MLPERF_SCRATCH_PATH/data/BraTS/MICCAI_BraTS_2019_Data_Training directory.
Run the following commands:

```bash
make download_model BENCHMARKS=3d-unet
make preprocess_data BENCHMARKS=3d-unet
```

### **(DOCKER)** Preprocessing

### **(DOCKER)** Generate the TensorRT engines:

```bash
# generates the TRT engines with the specified config. In this case it generates engine for both Offline and Server scenario and for default and high accuracy targets.
make generate_engines RUN_ARGS="--benchmarks=3d-unet --scenarios=Offline --config_ver=default,high_accuracy"
```

### **(DOCKER)** Run the benchmark:

```bash
# run the performance benchmark
make run_harness RUN_ARGS="--benchmarks=3d-unet --scenarios=Offline --config_ver=default --test_mode=PerformanceOnly"
make run_harness RUN_ARGS="--benchmarks=3d-unet --scenarios=Offline --config_ver=high_accuracy --test_mode=PerformanceOnly"

# run the accuracy benchmark
make run_harness RUN_ARGS="--benchmarks=3d-unet --scenarios=Offline --config_ver=default --test_mode=AccuracyOnly"
make run_harness RUN_ARGS="--benchmarks=3d-unet --scenarios=Offline --config_ver=high_accuracy --test_mode=AccuracyOnly"
```

## Limitations and Best Practices for Running MLPerf

Note the following limitations and best practices:

```bash
make run RUN_ARGS...
    is equal to
make generate_engines ... && make run_harness ...
```

- Fast build: use `--fast`

```bash
make run_harness RUN_ARGS="–-fast --benchmarks=<bmname> --scenarios=<scenario> --config_ver=<cver> --test_mode=PerformanceOnly"
```

The benchmark runs for one minute instead of the default 10 minutes.

If the server results are “INVALID”, reduce the server_target_qps for a Server scenario run. If the latency constraints are not met during the run, “INVALID” results are expected.
If the results are “INVALID” for an Offline scenario run, then increase the gpu_offline_expected_qps. “INVALID” runs for Offline scenario occur when the system can deliver a significantly higher QPS than what is provided through the gpu_offline_expected_qps configuration.
If the batch size changes, rebuild the engine.
Only the BERT, DLRM, 3D-Unet benchmarks support high accuracy targets.
3D-UNet only has Offline scenario.
