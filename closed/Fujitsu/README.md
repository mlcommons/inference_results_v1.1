README.md# MLPerf Inference v1.1 Implementations for Fujitsu Servers


## Benchmarks

The following *benchmarks* are part of our submission for MLPerf Inference:

 - **ResNet50** (resnet50)
 - **SSD-ResNet34** (ssd-resnet34)
 - **3D-Unet** (3d-unet)
 - **RNN-T** (rnn-t)
 - **BERT** (bert)
 - **DLRM** (dlrm)

## Scenarios
 - **Server**

## Fujitsu Submissions

| Benchmark     | Datacenter Submissions                                        |
|---------------|---------------------------------------------------------------|
| ResNet50      | Accuracy: 99% of FP32<br>Scenarios: Offline, Server           |
| SSD-ResNet34  | Accuracy: 99% of FP32<br>Scenarios: Offline, Server           |
| 3D-Unet       | Accuracy: 99% and 99.9% of FP32<br>Scenarios: Offline         |
| RNN-T         | Accuracy: 99% of FP32<br>Scenarios: Offline, Server           |
| BERT          | Accuracy: 99% and 99.9% of FP32<br>Scenarios: Offline, Server |
| DLRM          | Accuracy: 99% and 99.9% of FP32<br>Scenarios: Offline, Server |

Benchmarks are stored in the [code/](code) directory.
Every benchmark contains a `README.md` detailing instructions on how to set up that benchmark, including:

 - Downloading the dataset and model
 - Running any necessary preprocessing
 - Details on the optimizations being performed

For details on how to run each benchmark, see below.

## Fujitsu Submission Systems

The systems that Fujitsu supports, has tested, and are submitting are:

 - Datacenter system
   - A30x4 (Fujitsu PRIMERGY GX2460 m1)
   - A30x2 (Fujitsu PRIMERGY GX2460 m1)

## General Instructions


**Note:** Inside the Docker container, [closed/Fujitsu](closed/Fujitsu) will be mounted at `/work`.

Please refer to later sections for instructions on auditing.

We recommend using Ubuntu 20.04.
Other operating systems have not been tested.

### Before you run commands

Before running any commands detailed below, such as downloading and preprocessing datasets, or running any benchmarks, you should
set up the environment by doing the following:

- Run `export MLPERF_SCRATCH_PATH=<path/to/scratch/space>` to set your scratch space path.
We recommend that the scratch space has at least **3TB**.
The scratch space will be used to store models, datasets, and preprocessed datasets.

- For x86_64 systems (not Xavier): Run `make prebuild`.
This launches the Docker container with all the necessary packages installed.

 The docker image will have the tag `mlperf-inference:<USERNAME>-latest`.
 The source codes in the repository are located at `/work` inside the docker image.


For example, if you have a 8x A30 system, but you only wish to use 4, you can use:

```
make prebuild DOCKER_ARGS="--gpus '\"device=0,1,2,3\"'"
```

### Download and Preprocess Datasets and Download Models



```
$ make download_model # Downloads models and saves to $MLPERF_SCRATCH_PATH/models
Notes:

- The combined preprocessed data can be huge.

### Running the repository

#### Build

Builds the required libraries and TensorRT plugins:

```
See [command_flags.md](command_flags.md) for information on arguments that can be used with `RUN_ARGS`.
The optimized engine files are saved to `/work/build/engines`.

:warning: **IMPORTANT**: The DLRM harness requires around **40GB** of free CPU memory to load the dataset.
Otherwise, running the harness will crash with `std::bad_alloc`. :warning:

10 min. As this is quite the large increase in runtime duration, there is now a new `--fast` flag that can be specified
in `RUN_ARGS` that is a shortcut to specify `--min_duration=60000`. In Offline and MultiStream scenarios, this also sets
`--min_query_count=1`.

```
```

If `RUN_ARGS` is not specified, all harnesses for each supported benchmark-scenario pair will be run.
See [command_flags.md](command_flags.md) for `RUN_ARGS` options.

The performance results will be printed to `stdout`.
Other logging will be sent to `stderr`.
LoadGen logs can be found in `/work/build/logs`.

### Notes on runtime and performance

- To achieve maximum performance for Server scenario, please set Transparent Huge Pages (THP) to *always*.

### Run code in Headless mode

If you would like to run the repository without launching the interactive docker container, follow the steps below:

- `make build_docker NO_BUILD=1` to build the docker image.
- `make docker_add_user` to create a user in the docker image. (Skip this if you run as root.)
  - `make launch_docker DOCKER_COMMAND='make build'`
  - `make launch_docker DOCKER_COMMAND='make generate_engines RUN_ARGS="--benchmarks=resnet50 --scenarios=Offline"'`


```
$ make calibrate RUN_ARGS="--benchmarks=<BENCHMARK>"
```

See [calibration.md](calibration.md) for an explanation on how calibration is used for Fujitsu's submission.

### Update Results

Run the following command to update the LoadGen logs in `results/` with the logs in `build/logs`:

```
$ make update_results
```

Please refer to [submission_guide.md](submission_guide.md) for more detail about how to populate the logs requested by
MLPerf Inference rules under `results/`.

:warning: **IMPORTANT**: MLPerf Inference policies now have an option to allow submitters to submit an encrypted tarball of their
submission repository, and share a SHA1 of the encrypted tarball as well as the decryption password with the MLPerf
Inference results chair. This option gives submitters a more secure, private submission process.

:warning: For instructions on how to encrypt your submission, see the `Encrypting your project for submission` section
of [submission_guide.md](submission_guide.md).

### Run Compliance Tests and Update Compliance Logs

Please refer to [submission_guide.md](submission_guide.md).

## Instructions for Auditors

Please refer to the `README.md` in each benchmark directory for auditing instructions.

## Other documentations:

- [FAQ.md](FAQ.md): Frequently asked questions.
- [performance_tuning_guide.md](performance_tuning_guide.md): Instructions about how to run the benchmarks on your systems using our code base, solve the INVALID result issues, and tune the parameters for better performance.
- [submission_guide.md](submission_guide.md): Instructions about the required steps for a valid MLPerf Inference submission with or without our code base.
- [calibration.md](calibration.md): Instructions about how we did post-training quantization on activations and weights.
- [command_flags.md](command_flags.md): List of some flags supported by `RUN_ARGS`.
- [Per-benchmark READMEs](code/README.md): Instructions about how to download and preprocess the models and the datasets for each benchmarks and lists of optimizations we did for each benchmark.

