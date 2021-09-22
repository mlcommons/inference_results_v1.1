# syntax = docker/dockerfile:experimental
# based onhttps://github.com/pytorch/pytorch/blob/master/Dockerfile
# 
# NOTE: To build this you will need a docker version > 18.06 with
#       experimental enabled and DOCKER_BUILDKIT=1
#
#       If you do not use buildkit you are not going to have a good time
#
#       For reference: 
#           https://docs.docker.com/develop/develop-images/build_enhancements/

ARG BASE_IMAGE=ubuntu:20.04
FROM ${BASE_IMAGE} AS dev-base
RUN --mount=type=cache,id=apt-dev,target=/var/cache/apt \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    ca-certificates \
    git \
    curl \
    vim \
    build-essential \
    ccache \
    numactl \
    cmake \
    libjpeg-dev \
    libpng-dev \
    pybind11-dev \
    sudo \
    wget \
    libunwind8 \
    && rm -rf /var/lib/apt/lists/*
RUN /usr/sbin/update-ccache-symlinks
RUN mkdir /opt/ccache && ccache --set-config=cache_dir=/opt/ccache
ENV PATH /opt/conda/bin:$PATH

FROM dev-base as conda
ARG PYTHON_VERSION=3.7
RUN curl -fsSL -v -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} conda-build ipython ninja cython typing pybind11 Pillow && \
    /opt/conda/bin/conda install -y ninja pyyaml setuptools cmake cffi typing_extensions && \
    /opt/conda/bin/conda install -y intel-openmp mkl mkl-include numpy --no-update-deps && \
    /opt/conda/bin/conda clean -ya

FROM dev-base AS build
COPY --from=conda /opt/conda /opt/conda
ARG TORCHVISION_VERSION=v0.6.0
ARG IPEX_VERSION=mlperf/dlrm/inference-1.1
ARG PYTORCH_VERSION=v1.7.0
RUN --mount=type=cache,target=/opt/ccache \
    wget https://github.com/gperftools/gperftools/releases/download/gperftools-2.9.1/gperftools-2.9.1.tar.gz && \
    tar -xvzf gperftools-2.9.1.tar.gz && cd gperftools-2.9.1 && ./configure --prefix=${CONDA_PREFIX:-"$(dirname $(which conda))/../"} && \
    make && make install && cd .. && \
    pip install lark-parser hypothesis && \
    git clone --single-branch --branch=${IPEX_VERSION} https://github.com/intel/intel-extension-for-pytorch intel-extension-for-pytorch && \
    cd intel-extension-for-pytorch && \
    git submodule sync && git submodule update --init --recursive && \
    git clone --single-branch --branch=${PYTORCH_VERSION} https://github.com/pytorch/pytorch && \
    cd pytorch && git submodule sync && \
    git submodule update --init --recursive && \
    git apply ../torch_patches/xpu-1.7.patch && \
    python setup.py install && \
    cd .. && python setup.py install

FROM dev-base as mp
COPY --from=build /opt/conda /opt/conda
WORKDIR /opt/workdir
COPY ./intel_inference_datacenter_v1-1 intel_inference_datacenter_v1-1
RUN pip install opencv-python absl-py opencv-python-headless intel-openmp && \
    git clone --recurse-submodules https://github.com/mlcommons/inference.git mlperf_inference && \
    cd mlperf_inference && git checkout r1.1 && \
    git submodule update --init --recursive && cd loadgen && \
    CFLAGS="-std=c++14" python setup.py install && cd .. && \
    pip install -e git+https://github.com/mlperf/logging@1.0.0-rc4#egg=mlperf-logging && cd .. && \
    pip install sklearn onnx tqdm

# ENV MALLOC_CONF "oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"
ENV LD_PRELOAD "/opt/conda/lib/libiomp5.so"
ENV LD_LIBRARY_PATH "/opt/conda/lib/python3.7/site-packages/lib/":"/opt/conda/lib/python3.7/site-packages/torch_ipex-1.2.0-py3.7-linux-x86_64.egg/lib":${LD_LIBRARY_PATH}
ENV CONDA_PREFIX "/opt/conda"
