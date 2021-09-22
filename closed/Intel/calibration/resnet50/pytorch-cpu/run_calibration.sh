##############################################################################
#### 1) int8 calibration step(fusion path using ipex):
####    bash run_accuracy_ipex.sh resnet50 $DATA_PATH $MODEL_PATH dnnl int8 jit resnet50_configure_jit_intel.json calibration
###############################################################################
export DNNL_PRIMITIVE_CACHE_CAPACITY=1024

BATCH_SIZE=128

CONFIG_FILE=""

ARGS=""
ARGS="$ARGS $1"
echo "###### running $1 model #######"
if [ "$1" == "resnext101_32x16d_swsl" ]; then
    ARGS="$ARGS --hub"
fi

ARGS="$ARGS $2"
echo "### dataset path: $2"

ARGS="$ARGS $3"
echo "### model path: $3"

if [ "$4" == "dnnl" ]; then
    ARGS="$ARGS --dnnl"
    echo "### running auto_dnnl mode"
fi

if [ "$5" == "int8" ]; then
    ARGS="$ARGS --int8"
    CONFIG_FILE="$CONFIG_FILE --configure-dir $7"
    echo "### running int8 datatype"
elif [ "$5" == "bf16" ]; then
    ARGS="$ARGS --bf16"
    echo "### running bf16 datatype"
else
    echo "### running fp32 datatype"
fi

if [ "$6" == "jit" ]; then
    ARGS="$ARGS --jit"
    echo "### running jit fusion path"
else
    echo "### running not jit fusion path"
fi

if [ "$8" == "calibration" ]; then
    BATCH_SIZE=1
    ARGS="$ARGS --calibration"
    echo "### running int8 calibration"
fi


CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
TOTAL_CORES=`expr $CORES \* $SOCKETS`

KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"


export OMP_NUM_THREADS=$TOTAL_CORES
export $KMP_SETTING

echo -e "### using OMP_NUM_THREADS=$TOTAL_CORES"
echo -e "### using $KMP_SETTING\n\n"
sleep 3

python -u main.py -e -a $ARGS --ipex --pretrained -j 0 -b $BATCH_SIZE $CONFIG_FILE
