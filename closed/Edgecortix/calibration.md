# MLPerf Inference v1.1 - Calibration

## PyTorch models
Edgecortix quantize and calibrate FP32 precision PyTorch models using PyTorch's built-in post-training static quantization framework.
Weights are quantized per-channel to 8 bit precision int8_t and activations to 8 bit precision uint8_t using the FBGemm quantization back-end.

Calibration script [calibrate_pytorch_model.py](calibrate_pytorch_model.py).

For more information about Pytorch post-trainning static quantization please refer to this [document](https://pytorch.org/blog/introduction-to-quantization-on-pytorch/#post-training-static-quantization) for a more detailed explanation.

## TensorFlow models
Edgecortix quantize and calibrate FP32 precision TensorFlow models using TensorFlow's built-in post-training full integer quantization method.
Weights are quantized per-channel to 8 bit precision int8_t and activations to 8 bit precision int8_t.

Calibration script [calibrate_tensorflow_model.py](calibrate_tensorflow_model.py).

For more information about TensorFlow post-trainning static quantization please refer to this [document](https://www.tensorflow.org/lite/performance/post_training_quantization) for a more detailed explanation.

