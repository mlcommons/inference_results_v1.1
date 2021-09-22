echo "Install dependency packages"
pip install -e git+https://github.com/mlperf/logging@1.0.0-rc4#egg=mlperf-logging
pip install future numpy pyyaml requests setuptools six typing_extensions dataclasses #pytorch requirement
pip install lark-parser absl-py tqdm
conda install cmake --yes
conda install intel-openmp mkl mkl-include numpy --no-update-deps --yes

echo "Install gperftools-2.9 for tcmalloc"
wget https://github.com/gperftools/gperftools/releases/download/gperftools-2.9.1/gperftools-2.9.1.tar.gz
tar -xvzf gperftools-2.9.1.tar.gz
cd gperftools-2.9.1
./configure --prefix=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
make && make install
