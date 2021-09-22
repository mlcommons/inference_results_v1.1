# Reproducibility report: system setup

## Install system dependencies

Similar to RPi4:
* https://github.com/mlcommons/ck/blob/master/docs/mlperf-automation/platform/rpi4-ubuntu.md

## Check python version (3.7+)

```bash
python3 --version

> 3.7.10
```

## Install [CK automation framework](https://github.com/mlcommons/ck)

Tested with the version 2.5.8

```bash
python3 -m pip install ck -U
```

*You may need to restart your bash to initialize PATH to CK scripts.*

## Prepare virtual environment with common MLPerf components

```bash
ck pull repo:mlcommons@ck-venv --checkout=mlperf-1.1-mlcommons
ck create venv:mlperf-inference --template=mlperf-inference-1.1-tvm-mlcommons
```

## Activate virtual environment

```bash
ck activate venv:mlperf-inference
```

## Install CK repository with ML/MLPerf automations 

```bash
ck where repo:mlcommons@ck-mlops
ck pull repo:ck-mlperf-inference
```

## Install CK components

```bash
ck install package --tags=lib,python-package,onnxruntime-cpu,1.8.1
ck install package --tags=lib,python-package,onnx,1.10.1

ck install package --tags=lib,python-package,pytorch,cpu,1.9.0
ck install package --tags=lib,python-package,torchvision,cpu,0.10.0

ck install package --tags=tool,cmake,prebuilt,v3.21.1
ck install package --tags=compiler,llvm,prebuilt,v12.0.0

ck install package --tags=compiler,tvm,src,dev-0.8-dnnl-int8-v2-mlperf-1.1 \
      --env.USE_OPENMP=gnu --j=4 --quiet
```

## Prepare image classification task

```
ck install package --tags=imagenet,2012,aux
```

### Plug local ImageNet into CK

Note that the ImageNet validation dataset is not available for public download. 
Please follow these [instructions](https://github.com/mlcommons/ck/blob/master/docs/mlperf-automation/datasets/imagenet2012.md) to obtain it.

Find the path with the ImageNet validation set (50,000 images) and plug it into CK as follows:

```bash
ck detect soft:dataset.imagenet.val --force_version=2012 --extra_tags=full \
      --search_dir={directory where the ImageNet val dataset is downloaded}
```

### See all installed packages and detected components

```bash
ck show env
ck show env --tags=tvm
ck show env --tags=mlcommons,src
ck locate env --tags=tvm
```




## Resources

Our on-going collaboration with MLCommons&trade; to make 
the MLPerf&trade; inference benchmark more customizable, portable and easy to use:

* [MLCommons working groups](https://mlcommons.org/en/groups)
* [Open-source CK framework](https://github.com/mlcommons/ck) and [MLCube](https://github.com/mlcommons/mlcube)
  * [ML/AI packages from the community, OctoML and the cTuning foundation](https://github.com/mlcommons/ck-mlops/tree/main/package)
  * [ML/AI workflows from the community, OctoML and the cTuning foundation](https://github.com/mlcommons/ck-mlops/tree/main/program)
* [Documentation for the CK-powered MLPerf automation suite](https://github.com/mlcommons/ck/tree/master/docs/mlperf-automation)
* [Prototype of the end-to-end submission workflow for the MLPerf inference benchmark](https://github.com/mlcommons/ck-mlops/tree/main/module/bench.mlperf.inference)
