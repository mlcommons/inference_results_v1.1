#!/usr/bin/env bash
export WORKDIR=`pwd`
REPODIR=$WORKDIR/frameworks.ai.benchmarking.mlperf.develop.inference-datacenter

PATTERN='[-a-zA-Z0-9_]*='
if [ $# -lt "0" ] ; then
    echo 'ERROR:'
    printf 'Please use following parameters:
    --code=<mlperf workload repo directory> 
    '
    exit 1
fi

for i in "$@"
do
    case $i in
        --code=*)
            code=`echo $i | sed "s/${PATTERN}//"`;;
        *)
            echo "Parameter $i not recognized."; exit 1;;
    esac
done

if [ -d $code ];then
    REPODIR=$code
fi

echo "Installiing jemalloc"
git clone  https://github.com/jemalloc/jemalloc.git jemalloc_source  
cd jemalloc_source
git checkout c8209150f9d219a137412b06431c9d52839c7272
./autogen.sh
./configure --prefix=$WORKDIR
make
make install

echo "Installiing dependencies for Resnet50"
pip install Pillow pycocotools==2.0.2
pip install sklearn onnx
pip install dataclasses
pip install opencv-python
pip install absl-py
conda install typing_extensions --yes
conda config --add channels intel
conda install ninja pyyaml setuptools cmake cffi typing intel-openmp --yes
conda install mkl mkl-include numpy -c intel --no-update-deps --yes

echo "Installiing torch vision"
cd ..
git clone https://github.com/pytorch/vision
cd vision
git checkout v0.6.0
python setup.py install
cd ..