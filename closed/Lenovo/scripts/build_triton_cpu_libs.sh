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

# This script will build the Tensorflow and OpenVino backends for Triton necessary to run
# the Triton-CPU benchmarks

# Set up build directory
build_dir="build/tmp/tritonbuild"
build_branch=mlperf-inference-v1.1
mkdir -p ${build_dir}
pushd ${build_dir}
rm -fr *

# Clone Triton server repository
git clone --single-branch --depth=1 -b ${build_branch} https://github.com/triton-inference-server/s\
erver.git

# Build for CPU, TensorFlow2 and OpenVINO backends only
rm -fr build
mkdir build
cd server

# Note that openvino_21_02 and openvino_21_04 backends are available only for
# Triton's mlperf-inference-v1.1 branch. Other build_branchs will not work with
# this signature. The more general signature is --backend=openvino:$(build_branch)
./build.py -v --build-dir=${build_dir}/build --cmake-dir=/w\
orkspace/build --repo-tag=common:${build_branch} --repo-tag=core:${\
build_branch} --repo-tag=backend:${build_branch} --repo-tag=thirdparty:${build_branch} --endpoint=g\
rpc --endpoint=http --backend=openvino_21_02:${build_branch} --backend=openvino_21_04:${build_branch}

# Copy out built libraries to host
popd
mkdir -p prebuilt_triton_libs
id=$(docker create tritonserver:latest)
if [ ! -d "openvino_21_02" ] 
then
	docker cp $id:/opt/tritonserver/backends/openvino_21_02 openvino_21_02
fi

if [ ! -d "openvino_21_04" ] 
then
	docker cp $id:/opt/tritonserver/backends/openvino_21_04 openvino_21_04
fi
docker cp $id:/opt/tritonserver/lib/libtritonserver.so prebuilt_triton_libs
docker rm -v $id
