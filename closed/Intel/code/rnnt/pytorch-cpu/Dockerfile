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
    pybind11-dev \
    libpng-dev \
    unzip \
    libsndfile1 \
    wget \
    sudo \
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
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} conda-build ipython cython typing pybind11 Pillow && \
    /opt/conda/bin/conda install -y ninja pyyaml setuptools cmake cffi typing pandas jemalloc=5.0.1 && \
    /opt/conda/bin/conda install -y intel-openmp mkl mkl-include numpy --no-update-deps && \
    /opt/conda/bin/conda install -c conda-forge gperftools && \
    /opt/conda/bin/conda clean -ya

FROM dev-base AS build
COPY --from=conda /opt/conda /opt/conda
ARG IPEX_VERSION=mlperf/inference-1.1
ARG PYTORCH_VERSION=v1.8.0
ARG TORCHVISION_VERSION=v0.9.1
RUN --mount=type=cache,target=/opt/ccache \
    pip install lark-parser hypothesis && \
    git clone --single-branch --branch=${IPEX_VERSION} https://github.com/intel/intel-extension-for-pytorch intel-extension-for-pytorch && \
    cd intel-extension-for-pytorch && \
    git submodule sync && git submodule update --init --recursive && \
    git clone --single-branch --branch=${PYTORCH_VERSION} https://github.com/pytorch/pytorch && \
    cd pytorch && git submodule sync && \
    git submodule update --init --recursive && \
    pip install -r requirements.txt && \
    python setup.py install && \
    cd .. && python setup.py install && \
    git clone --single-branch --branch=${TORCHVISION_VERSION} https://github.com/pytorch/vision && cd vision && \
    python setup.py install

FROM dev-base as mp
COPY --from=build /opt/conda /opt/conda
WORKDIR /opt/workdir
COPY ./intel_inference_datacenter_v1-1 intel_inference_datacenter_v1-1
RUN mkdir third_party && wget https://ftp.osuosl.org/pub/xiph/releases/flac/flac-1.3.2.tar.xz -O third_party/flac-1.3.2.tar.xz && \
    cd third_party && tar xf flac-1.3.2.tar.xz && mkdir local && export install_dir=/opt/workdir/third_party/local && \
    cd flac-1.3.2 && ./configure --prefix=$install_dir && make && make install && rm -rf ../flac-1.3.2.tar.xz && \
    cd .. && wget https://sourceforge.net/projects/sox/files/sox/14.4.2/sox-14.4.2.tar.gz -O ./sox-14.4.2.tar.gz && \
    tar zxf sox-14.4.2.tar.gz && cd sox-14.4.2 && \
    LDFLAGS="-L${install_dir}/lib" CFLAGS="-I${install_dir}/include" ./configure --prefix=$install_dir --with-flac && make && make install && \
    rm -rf ../sox-14.4.2.tar.gz && cd ../.. && \
    pip install opencv-python absl-py opencv-python-headless intel-openmp && \
    git clone --recurse-submodules https://github.com/mlcommons/inference.git mlperf_inference && \
    cd mlperf_inference && git checkout r1.1 && \
    git submodule update --init --recursive && cd loadgen && \
    CFLAGS="-std=c++14" python setup.py install && cd .. && \
    pip install -e git+https://github.com/mlperf/logging@0.7.0-rc2#egg=mlperf-logging && cd .. && \
    pip install toml unidecode inflect librosa sklearn onnx tqdm sox

# ENV MALLOC_CONF "oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"
ENV LD_PRELOAD "/opt/conda/lib/libiomp5.so"
ENV LD_LIBRARY_PATH "/opt/conda/lib/python3.7/site-packages/lib/":"/opt/conda/lib/python3.7/site-packages/torch_ipex-1.8.0-py3.7-linux-x86_64.egg/lib":${LD_LIBRARY_PATH}
ENV CONDA_PREFIX "/opt/conda"
