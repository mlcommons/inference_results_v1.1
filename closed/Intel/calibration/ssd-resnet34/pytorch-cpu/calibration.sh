#!/bin/bash

export DNNL_PRIMITIVE_CACHE_CAPACITY=1024
export USE_IPEX=1

if [ "x$DATA_DIR" == "x"  ]; then
    echo "DATA_DIR not set" && exit 1
fi
if [ "x$MODEL_DIR" == "x"  ]; then
    echo "MODEL_DIR not set" && exit 1
fi

BATCH_SIZE=32

CONFIG_FILE=""
ARGS=""
if [ "$1" == "int8" ]; then
    ARGS="$ARGS --int8"
    CONFIG_FILE="$CONFIG_FILE --configure-dir $3"
    echo "### running int8 datatype"
else
    echo "### running fp32 datatype"
fi

if [ "$2" == "jit" ]; then
    ARGS="$ARGS --jit"
    echo "### running jit fusion path"
else
    echo "### running native path"
fi

if [ "$4" == "calibration"  ]; then
    ARGS="$ARGS --calibration"
    echo "### running int8 calibration"
fi

if [ -n "$5" ]; then
    ARGS="$ARGS --iter $5"
fi

python infer.py --seed 1 --threshold 0.2 -b $BATCH_SIZE -j 0 --data $DATA_DIR --device 0 --checkpoint $MODEL_DIR --no-cuda --ipex $ARGS $CONFIG_FILE
