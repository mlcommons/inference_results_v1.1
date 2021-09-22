# MLPerf Inference
## Install python packages
```bash
pip install -r requirements.txt
```
## (Optional) Setup libnux for furiosa npu runtime
```bash
bash setup_libnux.sh
```
## Prepare dataset and model
```bash
python prepare.py
```
## Evaluate mlcommons v1.1 vision models
### ResNet50
```bash
python python/main.py --profile resnet50-onnxruntime \
                      --dataset-path assets/ilsvrc/Data/CLS-LOC/val \
                      --model assets/models/mlcommons_resnet50_v1.5.onnx
```
### ResNet50 fake quant
```bash
python python/main.py --profile ssd-resnet-golden \
                      --dataset-path assets/ilsvrc/Data/CLS-LOC/val \
                      --model assets/models/mlcommons_resnet50_v1.5_fake_quant.onnx
```
### ResNet50 int8
(legacy)
```bash
python python/main.py --profile resnet-golden-npu-legacy \
                      --dataset-path assets/ilsvrc/Data/CLS-LOC/val \
                      --model assets/models/legacy/mlcommons_resnet50_v1.5_int8_legacy.onnx
```

### SSD MobilenetV1
```bash
python python/main.py --profile ssd-mobilenet-golden \
                      --dataset-path assets/mscoco/upscale300 \
                      --model assets/models/mlcommons_ssd_mobilenet_v1.onnx
```
### SSD MobilenetV1 fake quant
```bash
python python/main.py --profile ssd-mobilenet-golden \
                      --dataset-path assets/mscoco/upscale300 \
                      --model assets/models/mlcommons_ssd_mobilenet_v1_fake_quant.onnx
```
### SSD MobilenetV1 int8
(legacy)
```bash
python python/main.py --profile ssd-mobilenet-golden-npu-legacy \
                      --dataset-path assets/mscoco/upscale300 \
                      --model assets/models/legacy/mlcommons_ssd_mobilenet_v1_int8_legacy.onnx
```
### SSD ResNet34
```bash
python python/main.py --profile ssd-resnet34-onnxruntime-tf \
                      --dataset-path assets/mscoco/upscale1200 \
                      --model assets/models/mlcommons_ssd_resnet34.onnx
```
### SSD ResNet34 fake quant
```bash
python python/main.py --profile ssd-resnet34-golden \
                      --dataset-path assets/mscoco/upscale1200 \
                      --model assets/models/mlcommons_ssd_resnet34_fake_quant.onnx
```
### SSD ResNet34 int8
(legacy)
```bash
python python/main.py --profile ssd-resnet34-golden-npu-legacy \
                      --dataset-path assets/mscoco/upscale1200 \
                      --model assets/models/legacy/mlcommons_ssd_resnet34_int8_legacy.onnx
```
## ONNXRuntime PTQ
### minmax calibration
```bash
python python/ort_quantization.py -i assets/models/experiment/resnet50_opset13.onnx \
                                  -o assets/models/experiment/resnet50_ortq_minmax.onnx \
                                  -d assets/calibration/imagenet_calibration
```
### entropy calibration
```bash
python python/ort_quantization.py -i assets/models/experiment/resnet50_opset13.onnx \
                                  -o assets/models/experiment/resnet50_ortq_entropy.onnx \
                                  -d assets/calibration/imagenet_calibration \
                                  --entropy-calibration
```