#!/bin/bash
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

sudo apt install -y python3.8 python3.8-dev \
 && sudo rm -rf /usr/bin/python3 \
 && sudo ln -s /usr/bin/python3.8 /usr/bin/python3 \
 && sudo rm -rf /usr/bin/python \
 && sudo ln -s /usr/bin/python3 /usr/bin/python \
 && sudo apt -y autoremove

sudo apt install -y cuda-toolkit-10.2
sudo apt install -y virtualenv moreutils libnuma-dev numactl sshpass
sudo apt install -y pkg-config zip g++ unzip zlib1g-dev ntpdate
sudo apt install -y --no-install-recommends clang libclang-dev libglib2.0-dev
sudo apt install -y libhdf5-serial-dev hdf5-tools libopenmpi2 default-jdk
sudo apt install -y zlib1g-dev zip libjpeg8-dev libhdf5-dev libtiff5-dev libffi-dev
sudo ln -s /usr/lib/aarch64-linux-gnu/libclang-6.0.so.1 /usr/lib/aarch64-linux-gnu/libclang.so

sudo apt install -y libssl-dev libfreetype6-dev libpng-dev # matplotlib dependencies
sudo apt install -y libatlas3-base libopenblas-base
sudo apt install -y git
sudo apt install -y git-lfs && git-lfs install

# Triton dependencies
sudo apt install -y autoconf automake build-essential libb64-dev libre2-dev \
    libcurl4-openssl-dev libtool libboost-dev rapidjson-dev patchelf \
    libopenblas-dev software-properties-common
sudo apt install -y --allow-downgrades libopencv-dev=3.2.0+dfsg-4ubuntu0.1 \
    libopencv-core-dev=3.2.0+dfsg-4ubuntu0.1

pushd /tmp

sudo apt remove -y cmake \
  && sudo rm -rf cmake-* \
  && wget https://cmake.org/files/v3.18/cmake-3.18.4.tar.gz \
  && tar -xf cmake-3.18.4.tar.gz \
  && cd cmake-3.18.4 && ./configure && sudo make -j2 install \
  && sudo ln -s /usr/local/bin/cmake /usr/bin/cmake \
  && cd /tmp && rm -rf cmake-*

popd

# CMake assumes that the CUDA toolkit is located in /usr/local/cuda
if [[ -e /usr/local/cuda/packages ]]; then sudo mv /usr/local/cuda /usr/local/cuda_packages && sudo ln -s /usr/local/cuda-10.2 /usr/local/cuda; fi

# Install other dependencies (PyTorch, TensorFlow, etc.)
export CUDA_ROOT=/usr/local/cuda-10.2
export CUDA_INC_DIR=$CUDA_ROOT/include
export PATH=$CUDA_ROOT/bin:/usr/bin:$PATH
export CPATH=$CUDA_ROOT/include:$CPATH
export LIBRARY_PATH=$CUDA_ROOT/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_ROOT/lib64:$LD_LIBRARY_PATH
sudo python3 -m pip install --upgrade setuptools wheel virtualenv \
 && sudo python3 -m pip install Cython==0.29.23 \
 && sudo python3 -m pip install numpy==1.18.5 \
 && sudo rm -rf /usr/lib/python3/dist-packages/yaml /usr/lib/python3/dist-packages/PyYAML* \
 && sudo python3 -m pip install -r scripts/requirements_xavier.txt \
 && sudo python3 -m pip install scipy==1.6.3 \
 && sudo python3 -m pip install matplotlib==3.4.2 pycocotools==2.0.2 scikit-learn==0.22.1 \
 && sudo -E python3 -m pip install pycuda==2021.1

pushd /tmp

# Install cub
sudo rm -rf cub-1.8.0.zip cub-1.8.0 /usr/include/aarch64-linux-gnu/cub \
 && wget https://github.com/NVlabs/cub/archive/1.8.0.zip -O cub-1.8.0.zip \
 && unzip cub-1.8.0.zip \
 && sudo mv cub-1.8.0/cub /usr/include/aarch64-linux-gnu/ \
 && sudo rm -rf cub-1.8.0.zip cub-1.8.0

# Install gflags
sudo rm -rf gflags \
 && git clone -b v2.2.1 https://github.com/gflags/gflags.git \
 && cd gflags \
 && mkdir build && cd build \
 && cmake -DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=ON -DBUILD_gflags_LIB=ON .. \
 && make -j \
 && sudo make install \
 && cd /tmp && sudo rm -rf gflags

# Install glog
sudo rm -rf glog \
 && git clone -b v0.3.5 https://github.com/google/glog.git \
 && cd glog \
 && cmake -H. -Bbuild -G "Unix Makefiles" -DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=ON \
 && cmake --build build \
 && sudo cmake --build build --target install \
 && cd /tmp && sudo rm -rf glog

# Install PyTorch 1.4.0
# Following https://forums.developer.nvidia.com/t/install-pytorch-with-python-3-8-on-jetpack-4-4-1/160060/10
sudo rm -rf pytorch \
 && git clone --recursive -b v1.4.0-py38-xavier http://github.com/nvpohanh/pytorch \
 && cd pytorch \
 && sudo python3 -m pip install -r requirements.txt \
 && USE_CUDA=0 USE_CUDNN=0 USE_MKLDNN=0 sudo -E python3 setup.py install \
 && sudo python3 -m pip install torchvision==0.2.2 \
 && cd /tmp \
 && sudo rm -rf pytorch

# Install TensorFlow 1.15.0
# Following https://github.com/jkjung-avt/jetson_nano/blob/master/install_tensorflow-1.15.0.sh
sudo rm -rf jetson_nano src /usr/local/bin/bazel \
 && git clone -b v1.15.0-py38-xavier https://github.com/nvpohanh/jetson_nano.git \
 && bash jetson_nano/install_bazel-0.26.1.sh \
 && bash jetson_nano/install_tensorflow-1.15.0.sh \
 && cd /tmp && sudo rm -rf jetson_nano src

popd

if [[ -e ../../scripts/install_xavier_dependencies_internal.sh ]]; then \
  bash ../../scripts/install_xavier_dependencies_internal.sh ; fi

pushd /tmp

# Install DALI 0.31.0, needed by RNN-T
rm -rf protobuf-cpp-3.11.1.tar.gz \
 && wget https://github.com/protocolbuffers/protobuf/releases/download/v3.11.1/protobuf-cpp-3.11.1.tar.gz \
 && tar -xzf protobuf-cpp-3.11.1.tar.gz \
 && rm protobuf-cpp-3.11.1.tar.gz \
 && cd protobuf-3.11.1 \
 && ./configure CXXFLAGS="-fPIC" --prefix=/usr/local --disable-shared \
 && make -j2 \
 && sudo make install \
 && sudo ldconfig \
 && cd /tmp \
 && rm -rf protobuf-3.11.1 \
 && cd /usr/local \
 && sudo rm -rf DALI \
 && sudo git clone -b release_v0.31 --recursive https://github.com/NVIDIA/DALI \
 && cd DALI \
 && sudo mkdir build \
 && cd build \
 && sudo cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCUDA_TARGET_ARCHS="72" \
    -DBUILD_PYTHON=ON -DBUILD_TEST=OFF -DBUILD_BENCHMARK=OFF -DBUILD_LMDB=OFF -DBUILD_NVTX=OFF -DBUILD_NVJPEG=OFF \
    -DBUILD_LIBTIFF=OFF -DBUILD_NVOF=OFF -DBUILD_NVDEC=OFF -DBUILD_LIBSND=OFF -DBUILD_NVML=OFF -DBUILD_FFTS=ON \
    -DVERBOSE_LOGS=OFF -DWERROR=OFF -DBUILD_WITH_ASAN=OFF .. \
 && sudo make -j2 \
 && sudo make install \
 && sudo python3 -m pip install dali/python/ \
 && sudo mv /usr/local/DALI/build/dali/python/nvidia/dali /tmp/dali \
 && sudo rm -rf /usr/local/DALI \
 && sudo mkdir -p /usr/local/DALI/build/dali/python/nvidia/ \
 && sudo mv /tmp/dali /usr/local/DALI/build/dali/python/nvidia/ \
 && cd /tmp

# Install ONNX graph surgeon, needed for 3D-Unet ONNX preprocessing.
cd /tmp \
 && rm -rf TensorRT \
 && git clone https://github.com/NVIDIA/TensorRT.git \
 && cd TensorRT \
 && git checkout release/7.2 \
 && cd tools/onnx-graphsurgeon \
 && make build \
 && sudo python3 -m pip install --no-deps -t /usr/local/lib/python3.8/dist-packages --force-reinstall dist/*.whl \
 && cd /tmp \
 && rm -rf TensorRT
