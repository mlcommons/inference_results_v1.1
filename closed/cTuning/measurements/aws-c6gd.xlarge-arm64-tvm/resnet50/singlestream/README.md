# Reproducibility report: benchmarking

Do not forget to prepare your system based on the "Reproducibility report: system setup".


## Install ONNX model (ResNet50; opset-11; FP32)

Install MLPerf model:

```bash
ck install package --tags=model,image-classification,mlperf,onnx,v1.5-opset-11
```

More information about this model: 
[ [CK meta.json](https://github.com/mlcommons/ck-mlops/blob/main/package/ml-model-mlperf-resnet50-onnx/.cm/meta.json) ]


## Install CK workflow Python dependencies

```bash
ck run program:mlperf-inference-bench-image-classification-tvm-onnx-cpu \
     --cmd_key=install-python-requirements
```


## Run Offline scenario

Since it's too long to run this scenario and we can't satisfy min number of samples (24K), we use SingleStream results.

You can still try to run offline mode as follows:

### Accuracy

```bash
time ck benchmark program:mlperf-inference-bench-image-classification-tvm-onnx-cpu \
     --repetitions=1 --skip_print_timers --skip_print_stats --skip_stat_analysis \
     --cmd_key=accuracy-offline \
     --env.MLPERF_BACKEND=tvm \
     --env.MLPERF_TVM_EXECUTOR=graph \
     --env.MLPERF_TVM_TARGET="llvm" \
     --env.OMP_NUM_THREADS=1 \
     --env.TVM_NUM_THREADS=1 \
     --env.EXTRA_OPS="--max-batchsize 1" \
     --print_files=accuracy.txt
```

Extra customization:
* ```--env.EXTRA_OPS="--threads 1 --max-batchsize 1 --count 100 --time 60")```
* ```llvm -mcpu=cortex-a72 -mfloat-abi=hard```


### Performance

```bash
time ck benchmark program:mlperf-inference-bench-image-classification-tvm-onnx-cpu \
     --repetitions=1 --skip_print_timers --skip_print_stats --skip_stat_analysis \
     --dep_add_tags.dataset=full \
     --cmd_key=performance-offline \
     --env.MLPERF_BACKEND=tvm \
     --env.MLPERF_TVM_EXECUTOR=graph \
     --env.MLPERF_TVM_TARGET="llvm" \
     --env.OMP_NUM_THREADS=1 \
     --env.TVM_NUM_THREADS=1 \
     --env.EXTRA_OPS="--max-batchsize 1 --qps 2 --count 1024" \
     --print_files=mlperf_log_summary.txt --no_clean
```

## SingleStream scenario

### Accuracy

```bash
time ck benchmark program:mlperf-inference-bench-image-classification-tvm-onnx-cpu \
     --repetitions=1 --skip_print_timers --skip_print_stats --skip_stat_analysis \
     --cmd_key=accuracy-singlestream \
     --env.MLPERF_BACKEND=tvm \
     --env.MLPERF_TVM_EXECUTOR=graph \
     --env.MLPERF_TVM_TARGET="llvm" \
     --env.OMP_NUM_THREADS=4 \
     --env.TVM_NUM_THREADS=4 \
     --env.EXTRA_OPS="--threads 1 --max-batchsize 1" \
     --print_files=accuracy.txt
```

### Performance

```bash
time ck benchmark program:mlperf-inference-bench-image-classification-tvm-onnx-cpu \
     --no_clean \
     --repetitions=1 --skip_print_timers --skip_print_stats --skip_stat_analysis \
     --cmd_key=performance-singlestream \
     --env.MLPERF_BACKEND=tvm \
     --env.MLPERF_TVM_EXECUTOR=graph \
     --env.MLPERF_TVM_TARGET="llvm" \
     --env.OMP_NUM_THREADS=4 \
     --env.TVM_NUM_THREADS=4 \
     --env.EXTRA_OPS="--threads 1 --max-batchsize 1 --count 8192" \
     --print_files=mlperf_log_summary.txt
```


*Add "--no_clean" flag to avoid recompiling TVM model*






## Prepare your submission

One can use the [end-to-end MLPerf submission and visualization workflow](https://github.com/mlcommons/ck-mlops/tree/main/module/bench.mlperf.inference)
(collaboration between MLCommons, OctoML and the cTuning foundation) to prepare the above submission as follows:

```bash

ck activate venv:mlperf-inference

ck pull repo:ck-mlperf-inference

ck install package --tags=mlperf,inference,results,dummy

ck set kernel --var.mlperf_inference_version=1.1
ck set kernel --var.mlperf_inference_submitter=cTuning
ck set kernel --var.mlperf_inference_division=closed

ck add bench.mlperf.system:aws-c6gd.xlarge-arm64-tvm --base=rpi4-tflite-v2.2.0-ruy
ck set kernel --var.mlperf_inference_system=aws-c6gd.xlarge-arm64-tvm


ck run bench.mlperf.inference  --framework=tvm-onnx --model=resnet50 --scenario=offline --mode=prereq


ck run bench.mlperf.inference  --framework=tvm-onnx --model=resnet50 --scenario=singlestream --mode=accuracy \
        --skip_system_ext \
        --clean \
        --duplicate_to_offline \
        --cmd_key=accuracy-singlestream \
        --env.MLPERF_BACKEND=tvm \
        --env.MLPERF_TVM_EXECUTOR=graph \
        --env.MLPERF_TVM_TARGET="llvm" \
        --env.OMP_NUM_THREADS=4 \
        --env.TVM_NUM_THREADS=4 \
        --env.EXTRA_OPS="--threads 1 --max-batchsize 1"

ck run bench.mlperf.inference  --framework=tvm-onnx --model=resnet50 --scenario=singlestream --mode=performance \
        --skip_system_ext \
        --clean \
        --duplicate_to_offline \
        --compliance --quiet \
        --cmd_key=accuracy-singlestream \
        --env.MLPERF_BACKEND=tvm \
        --env.MLPERF_TVM_EXECUTOR=graph \
        --env.MLPERF_TVM_TARGET="llvm" \
        --env.OMP_NUM_THREADS=4 \
        --env.TVM_NUM_THREADS=4 \
        --env.EXTRA_OPS="--threads 1 --max-batchsize 1 --count 8192"
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
