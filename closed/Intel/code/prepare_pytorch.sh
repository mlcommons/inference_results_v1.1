WORKDIR=$1
pushd .
cd $WORKDIR
echo Current directory is $PWD
echo Using gcc=`which gcc`
echo "GCC version should >= 9"
gcc --version
CC=`which gcc`
WORKDIR=$PWD

# install pytorch
echo "Install pytorch/ipex"
export LD_LIBRARY_PATH=$WORKDIR/local/lib:$LD_LIBRARY_PATH
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
export CMAKE_LIBRARY_PATH=${CMAKE_PREFIX_PATH}/lib
export CMAKE_INCLUDE_PATH=${CMAKE_PREFIX_PATH}/include

conda install astunparse numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses

cd $WORKDIR
git clone https://github.com/pytorch/pytorch.git
cd pytorch
git checkout v1.8.0
git log -1
git submodule sync
git submodule update --init --recursive
pip install -r requirements.txt
python setup.py install

cd $WORKDIR
git clone https://github.com/intel/intel-extension-for-pytorch intel-extension-for-pytorch
cd intel-extension-for-pytorch
git checkout mlperf/inference-1.1
git log -1
git submodule sync
git submodule update --init --recursive
pip install lark-parser hypothesis
python setup.py install

popd
