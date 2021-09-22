
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
    autoconf \
    libgl1 \
    libglib2.0-0 \
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
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} && \
    /opt/conda/bin/conda config --add channels intel && \
    /opt/conda/bin/conda install -y ninja pyyaml setuptools cmake cffi typing intel-openmp && \
    /opt/conda/bin/conda install -y mkl mkl-include numpy -c intel --no-update-deps && \
    /opt/conda/bin/conda install -y astunparse typing_extensions future six requests dataclasses && \
    /opt/conda/bin/conda clean -ya

FROM dev-base AS build
COPY --from=conda /opt/conda /opt/conda
ARG TORCHVISION_VERSION=v0.6.0
ARG IPEX_VERSION=mlperf/inference-1.1
ARG PYTORCH_VERSION=v1.8.0
RUN --mount=type=cache,target=/opt/ccache \
    git clone --single-branch --branch=${PYTORCH_VERSION} https://github.com/pytorch/pytorch && \
    cd pytorch && git submodule sync && \
    git submodule update --init --recursive && \
    pip install -r requirements.txt && python setup.py install && cd .. && \
    git clone --single-branch --branch=${IPEX_VERSION} https://github.com/intel/intel-extension-for-pytorch intel-extension-for-pytorch && \
    cd intel-extension-for-pytorch && \
    git submodule sync && git submodule update --init --recursive && \
    pip install lark-parser hypothesis && python setup.py install && \
    git clone --single-branch --branch=${TORCHVISION_VERSION} https://github.com/pytorch/vision && cd vision && \
    python setup.py install

FROM dev-base as mp
COPY --from=build /opt/conda /opt/conda
ARG JEMALLOC_COMMIT=c8209150f9d219a137412b06431c9d52839c7272
WORKDIR /opt/workdir
COPY ./intel_inference_datacenter_v1-1 intel_inference_datacenter_v1-1
RUN git clone https://github.com/jemalloc/jemalloc.git jemalloc_source && \
    cd jemalloc_source && git reset --hard ${JEMALLOC_COMMIT} && \
    bash autogen.sh && ./configure --prefix=/opt/workdir/jemalloc && make && make install && \
    cd .. && pip install Pillow pycocotools==2.0.2 sklearn onnx dataclasses opencv-python absl-py && \
    git clone --recurse-submodules https://github.com/mlcommons/inference.git mlperf_inference && \
    cd mlperf_inference && git checkout r1.1 && \
    git submodule update --init --recursive && cd loadgen && \
    CFLAGS="-std=c++14" python setup.py install

# ENV MALLOC_CONF "oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"
ENV LD_PRELOAD "/opt/conda/lib/libiomp5.so /opt/workdir/jemalloc/lib/libjemalloc.so"
ENV LD_LIBRARY_PATH "/opt/conda/lib/python3.7/site-packages/lib/":"/opt/conda/lib/python3.7/site-packages/torch_ipex-1.8.0-py3.7-linux-x86_64.egg/lib":${LD_LIBRARY_PATH}
ENV CONDA_PREFIX "/opt/conda"
