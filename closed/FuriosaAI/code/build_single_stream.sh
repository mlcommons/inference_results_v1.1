#!/bin/bash
export NPU_GLOBAL_CONFIG_PATH=../common/npu_config/128dpes.yml

cd resnet50
NPU_COMPILER_CONFIG_PATH=<(echo "{ remove_lower: true, remove_unlower: true }") ../common/compiler ./mlcommons_resnet50_v1.5_int8.onnx mlcommons_resnet50_v1.5_int8.enf
make clean
make
cd -

cd ssd-mobilenet
NPU_COMPILER_CONFIG_PATH=<(echo "{ remove_lower: true, remove_unlower: true }") ../common/compiler ./mlcommons_ssd_mobilenet_v1_int8.onnx ./mlcommons_ssd_mobilenet_v1_int8.enf
make clean
make
cd -

cd ssd-resnet34
NPU_COMPILER_CONFIG_PATH=<(echo "{ remove_lower: true, remove_unlower: true }") ../common/compiler ./mlcommons_ssd_resnet34_int8.onnx ./mlcommons_ssd_resnet34_int8.enf
make clean
make
cd -
