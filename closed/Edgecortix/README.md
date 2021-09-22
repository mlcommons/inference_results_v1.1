# MLPerf Inference v1.1 Edgecortix-Optimized Implementations

This folder contains Edgecortix's optimized implementations for [MLPerf Inference Benchmark v1.1](https://www.mlperf.org/inference-overview/) closed division.

## Benchmarks

The following *benchmarks* are part of our submission for MLPerf Inference v1.1:

 - **ResNet50** (resnet50)

## Scenarios

Each of the above benchmarks can be run in one or more of the following two inference *scenarios*:

 - **SingleStream**
 - **Offline**

Please refer to the [MLPerf Inference official page](https://www.mlperf.org/inference-overview/) for explanations about the scenarios.

## Edgecortix Submissions

Our MLPerf Inference v1.1 implementation has the following submissions:

| Benchmark     | Edge Submissions                                                                 |
|---------------|----------------------------------------------------------------------------------|
| ResNet50      | Accuracy: 99% of FP32<br>Scenarios: Offline, SingleStream                        |


## Edgecortix Submission Systems

The systems that Edgecortix supports and has tested are:

 - Edge systems
   - Dynamic Neural Accelerator (DNA)

## Usage

Here we show an usage example of how to run benchmarks in both performance and accuracy mode.
The whole deployment process:

 - download the FP32 precision model
 - preparation for quantization
 - calibration process
 - quantized model compilation and deployment
 - and finally, model execution

is all part of our additional backend, available [here](code/resnet50/SingleStream/python/backend_edgecortix.py).

This additional backend allows us to use the reference implementation inference scripts provided by MLPerf as usual. It is self-contained and reflects the whole deployment process. For convenience, a simple Python interface that allows executing an already-deployed model is also provided [here](code/resnet50/SingleStream/python/ip_runtime/ip_rt.py).

### Prepare the dataset

The validation dataset directory can be set using the environment variable `DATA_DIR`, in this example it points to the directory where the ImageNet validation dataset is located.

### Performance-only benchmark

First parameter `edgecortix` makes reference to the new backend provided by Edgecortix.
To run the benchmark in performance mode using, for example, the Resnet50-v1.5 PyTorch model provided by TorchVision package, we should specify `resnet50` as the second parameter.
An additional parameter is the file that contains the list of images used during the calibration stage. In our case, we choose the list number one provided by MLPerf.

```bash
cd vision/classification_and_detection
DATA_DIR=/opt/edgecortix/imagenet/ ./run_local.sh edgecortix resnet50 --dataset-calibration-list ../../calibration/ImageNet/cal_image_list_option_1.txt
```
### Accuracy benchmark

Similarly, to run in accuracy mode:

```bash
cd vision/classification_and_detection
DATA_DIR=/opt/edgecortix/imagenet/ ./run_local.sh edgecortix resnet50 --dataset-calibration-list ../../calibration/ImageNet/cal_image_list_option_1.txt --accuracy
```

## Quantization and calibration

### PyTorch models
Edgecortix quantize and calibrate FP32 precision PyTorch models using PyTorch's built-in post-training static quantization framework.
Weights are quantized per-channel to 8 bit precision int8_t and activations to 8 bit precision uint8_t using the FBGemm quantization back-end.

Calibration script [calibrate_pytorch_model.py](calibrate_pytorch_model.py).

For more information about Pytorch post-trainning static quantization please refer to this [document](https://pytorch.org/blog/introduction-to-quantization-on-pytorch/#post-training-static-quantization) for a more detailed explanation.

### TensorFlow models
Edgecortix quantize and calibrate FP32 precision TensorFlow models using TensorFlow's built-in post-training full integer quantization method.
Weights are quantized per-channel to 8 bit precision int8_t and activations to 8 bit precision int8_t.

Calibration script [calibrate_tensorflow_model.py](calibrate_tensorflow_model.py).

For more information about TensorFlow post-trainning static quantization please refer to this [document](https://www.tensorflow.org/lite/performance/post_training_quantization) for a more detailed explanation.
