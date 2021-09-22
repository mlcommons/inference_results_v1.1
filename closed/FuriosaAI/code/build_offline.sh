#!/bin/bash

cd resnet50
export NPU_GLOBAL_CONFIG_PATH=../common/npu_config/64dpes.yml
NPU_COMPILER_CONFIG_PATH=<(echo "{ remove_lower: true, remove_unlower: true }") ../common/compiler ./mlcommons_resnet50_v1.5_int8.onnx mlcommons_resnet50_v1.5_int8_batch8.enf -b 8
make clean
make
cd -

cd ssd-mobilenet
export NPU_GLOBAL_CONFIG_PATH=../common/npu_config/64dpes.yml
NPU_COMPILER_CONFIG_PATH=<(echo "{ remove_lower: true, remove_unlower: true, split_after_lower: true }") ../common/compiler ./mlcommons_ssd_mobilenet_v1_int8.onnx ./mlcommons_ssd_mobilenet_v1_int8_batch8.enf -b 8
make clean
make
cd -

cd ssd-resnet34
export NPU_GLOBAL_CONFIG_PATH=../common/npu_config/128dpes.yml
NPU_COMPILER_CONFIG_PATH=<(echo "{ remove_lower: true, remove_unlower: true }") ../common/compiler ./mlcommons_ssd_resnet34_int8.onnx ./mlcommons_ssd_resnet34_int8.enf
make clean
make
cd -
