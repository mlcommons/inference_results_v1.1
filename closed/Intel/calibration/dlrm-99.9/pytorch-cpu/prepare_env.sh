  export WORKDIR=`pwd`

  cd ${WORKDIR}
  # clone PyTorch
  git clone https://github.com/pytorch/pytorch.git
  cd pytorch && git checkout v1.7.0
  git log -1
  git submodule sync && git submodule update --init --recursive
  cd ..

  # clone Intel Extension for PyTorch
  #git clone https://gitlab.devtools.intel.com/intel-pytorch-extension/ipex-cpu-dev.git
  git clone https://github.com/intel/intel-extension-for-pytorch
  cd ipex-cpu-dev
  git checkout mlperf/dlrm/inference-1.1
  git log -1
  git submodule sync && git submodule update --init --recursive

  # install PyTorch first
  cd ${WORKDIR}/pytorch
  cp ${WORKDIR}/ipex-cpu-dev/torch_patches/xpu-1.7.patch .
  git apply xpu-1.7.patch
  python setup.py install

  # install Intel Extension for PyTorch
  cd ${WORKDIR}/ipex-cpu-dev
  python setup.py install
  cd ..
