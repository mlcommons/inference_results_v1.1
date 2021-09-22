# Reproducibility report: benchmarking

First, you must prepare your system based on the "Reproducibility report: system setup".


## Install TensorFlow model (ResNet50; FP32)

Install MLPerf model:

```bash
ck install package --tags=model,image-classification,mlperf,tensorflow,resnet50
```

More information about this model: 
[ [CK meta.json](https://github.com/mlcommons/ck-mlops/blob/main/package/ml-model-mlperf-resnet50-tf/.cm/meta.json) ]


## Install CK workflow Python dependencies

```bash
ck run program:mlperf-inference-bench-image-classification-tensorflow-cpu \
     --cmd_key=install-python-requirements
```

## Run Offline scenario

### Accuracy

```bash
time ck benchmark program:mlperf-inference-bench-image-classification-tensorflow-cpu \
     --repetitions=1 --skip_print_timers --skip_print_stats --skip_stat_analysis \
     --cmd_key=accuracy-offline \
     --env.EXTRA_OPS="" \
     --print_files=accuracy.txt
```

### Performance

```bash

time ck benchmark program:mlperf-inference-bench-image-classification-tensorflow-cpu \
     --repetitions=1 --skip_print_timers --skip_print_stats --skip_stat_analysis \
     --cmd_key=performance-offline \
     --env.EXTRA_OPS="--qps 150 --time 610" \
     --print_files=mlperf_log_summary.txt

```






## Prepare your submission

One can use the [end-to-end MLPerf submission and visualization workflow](https://github.com/mlcommons/ck-mlops/tree/main/module/bench.mlperf.inference)
(collaboration between MLCommons, OctoML and the cTuning foundation) to prepare the above submission as follows:

```bash

ck activate venv:mlperf-inference

ck pull repo:ck-mlperf-inference

ck install package --tags=mlperf,inference,results,dummy

ck set kernel --var.mlperf_inference_version=1.1
ck set kernel --var.mlperf_inference_submitter=cTuning
ck set kernel --var.mlperf_inference_division=open

ck add bench.mlperf.system:aws-m5zn.6xlarge-tf-mlperf-off-the-shelf --base=1-node-2s-clx-tf-int8
ck set kernel --var.mlperf_inference_system=aws-m5zn.6xlarge-tf-mlperf-off-the-shelf

ck run bench.mlperf.inference  --framework=tf --model=resnet50 --scenario=offline --mode=prereq

ck run bench.mlperf.inference  --framework=tf --model=resnet50 --scenario=offline --mode=accuracy \
     --skip_system_ext \
     --env.EXTRA_OPS=""

ck run bench.mlperf.inference  --framework=tf --model=resnet50 --scenario=offline --mode=performance \
     --skip_system_ext \
     --env.EXTRA_OPS="--qps 100 --time 610"
```

You should truncate your accuracy files before submitting results:
```bash
ck run program:mlperf-inference-submission --cmd_key=truncate_accuracy_log --env.CK_MLPERF_SUBMITTER=cTuning
```

Check the submission by the MLPerf submission checker:
```bash
ck run program:mlperf-inference-submission --cmd_key=check
```

Pack results:
```
ck zip bench.mlperf.system
```



## Visualize MLPerf results

* [Prototype of a local dashboard for the MLPerf inference benchmark](https://github.com/mlcommons/ck-mlops/blob/main/module/bench.mlperf.inference/README.results.md)




## Resources

Our on-going collaboration with MLCommons to make 
the MLPerf&trade; inference benchmark more customizable, portable and easy to use:

* [MLCommons working groups](https://mlcommons.org/en/groups)
* [Open-source CK framework](https://github.com/mlcommons/ck) and [MLCube](https://github.com/mlcommons/mlcube)
  * [ML/AI packages from the community, OctoML and the cTuning foundation](https://github.com/mlcommons/ck-mlops/tree/main/package)
  * [ML/AI workflows from the community, OctoML and the cTuning foundation](https://github.com/mlcommons/ck-mlops/tree/main/program)
* [Documentation for the CK-powered MLPerf automation suite](https://github.com/mlcommons/ck/tree/master/docs/mlperf-automation)
* [Prototype of the end-to-end submission workflow for the MLPerf inference benchmark](https://github.com/mlcommons/ck-mlops/tree/main/module/bench.mlperf.inference)
