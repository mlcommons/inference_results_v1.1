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
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} ninja cmake jemalloc tensorflow && \
    /opt/conda/bin/conda install -c conda-forge llvm-openmp && \
    /opt/conda/bin/conda install -c intel mkl mkl-include intel-openmp && \
    /opt/conda/bin/conda clean -ya

FROM dev-base AS build
COPY --from=conda /opt/conda /opt/conda
WORKDIR /opt/workdir
COPY ./intel_inference_datacenter_v1-1 intel_inference_datacenter_v1-1
ENV CONDA_PREFIX "/opt/conda"
ARG PYTORCH_VERSION=v1.9.0
ARG PYTORCH_COMMIT=3979cb0656fe2d8b0445768a769bd624b10778b5
ARG TRANSFORMERS_COMMIT=9f4e0c23d68366985f9f584388874477ad6472d8
RUN git clone https://github.com/pytorch/pytorch.git && \
    cd pytorch && git reset --hard ${PYTORCH_COMMIT} && \
    pip install -r requirements.txt && \
    git apply ../intel_inference_datacenter_v1-1/closed/Intel/code/bert-99/pytorch-cpu/patches/clean_openmp.patch && \
    git apply ../intel_inference_datacenter_v1-1/closed/Intel/code/bert-99/pytorch-cpu/patches/pytorch.patch && \
    USE_CUDA=OFF python -m pip install -e . && cd .. && \
    git clone https://github.com/huggingface/transformers.git && \
    cd transformers && git checkout ${TRANSFORMERS_COMMIT} && \ 
    git apply ../intel_inference_datacenter_v1-1/closed/Intel/code/bert-99/pytorch-cpu/patches/transformers.patch && \
    python -m pip install -e . 

FROM dev-base as mp
COPY --from=build /opt/conda /opt/conda
COPY --from=build /opt/workdir /opt/workdir
WORKDIR /opt/workdir
ENV CONDA_PREFIX "/opt/conda"
ARG ONEDNN_COMMIT=eac1e4a51a2c64129ec6475b8053b06463b667a9
RUN cd intel_inference_datacenter_v1-1/closed/Intel/code/bert-99/pytorch-cpu/ && \
    git clone --branch r1.1 --recurse https://github.com/mlcommons/inference.git && cd mlperf_plugins && \
    git clone https://github.com/oneapi-src/oneDNN.git onednn && cd onednn && \
    git reset --hard ${ONEDNN_COMMIT} && git apply ../../patches/onednn.patch && cd ../.. && \
    mkdir build && cd build && \
    cmake -DCMAKE_PREFIX_PATH="$(dirname $(python3 -c 'import torch; print(torch.__file__)'))" -GNinja -DCMAKE_BUILD_TYPE=Release .. && \
    ninja && pip install tokenization

ENV LD_LIBRARY_PATH "/opt/conda/lib/python3.7/site-packages/lib/"
ENV CONDA_PREFIX "/opt/conda"
