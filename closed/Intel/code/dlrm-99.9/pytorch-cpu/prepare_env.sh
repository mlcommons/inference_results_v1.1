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

  echo "Install loadgen"
  git clone https://github.com/mlcommons/inference.git
  cd inference && git checkout r1.1
  git log -1
  git submodule update --init --recursive
  cd loadgen
  CFLAGS="-std=c++14" python setup.py install
  cd ..; cp ${WORKDIR}/inference/mlperf.conf ${REPODIR}/closed/Intel/code/dlrm-99.9/pytorch-cpu/. 

  echo "Clone source code and Install"
  echo "Install PyTorch and Intel Extension for PyTorch"

  cd ${WORKDIR}
  # clone PyTorch
  git clone https://github.com/pytorch/pytorch.git
  cd pytorch && git checkout v1.7.0
  git log -1
  git submodule sync && git submodule update --init --recursive
  cd ..

  # clone Intel Extension for PyTorch
  git clone https://github.com/intel/intel-extension-for-pytorch ipex-cpu-dev
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
