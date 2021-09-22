echo "Install dependency packages"
pip install sklearn onnx tqdm lark-parser
pip install -e git+https://github.com/mlperf/logging@1.0.0-rc4#egg=mlperf-logging
conda install ninja pyyaml setuptools cmake cffi typing --yes
conda install intel-openmp mkl mkl-include numpy --no-update-deps --yes
pip install opencv-python  absl-py opencv-python-headless

echo "Install gperftools-2.9 for tcmalloc"
wget https://github.com/gperftools/gperftools/releases/download/gperftools-2.9.1/gperftools-2.9.1.tar.gz
tar -xvzf gperftools-2.9.1.tar.gz
cd gperftools-2.9.1
./configure --prefix=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
make && make install
