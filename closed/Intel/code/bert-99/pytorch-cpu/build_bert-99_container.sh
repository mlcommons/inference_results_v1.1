#!/bin/env bash

export DOCKER_BUILD_ARGS="--build-arg ftp_proxy=${ftp_proxy} --build-arg FTP_PROXY=${FTP_PROXY} --build-arg http_proxy=${http_proxy} --build-arg HTTP_PROXY=${HTTP_PROXY} --build-arg https_proxy=${https_proxy} --build-arg HTTPS_PROXY=${HTTPS_PROXY} --build-arg no_proxy=${no_proxy} --build-arg NO_PROXY=${NO_PROXY} --build-arg socks_proxy=${socks_proxy} --build-arg SOCKS_PROXY=${SOCKS_PROXY}"

export DOCKER_RUN_ENVS="--env ftp_proxy=${ftp_proxy} --env FTP_PROXY=${FTP_PROXY} --env http_proxy=${http_proxy} --env HTTP_PROXY=${HTTP_PROXY} --env https_proxy=${https_proxy} --env HTTPS_PROXY=${HTTPS_PROXY} --env no_proxy=${no_proxy} --env NO_PROXY=${NO_PROXY} --env socks_proxy=${socks_proxy} --env SOCKS_PROXY=${SOCKS_PROXY}"

export PYTORCH_VERSION=v1.9.0
export IMAGE_NAME=intel/intel-optimized-pytorch:${PYTORCH_VERSION}-bert
export SUBMISSIONS_INFERENCE_1_1_REPO="${SUBMISSIONS_INFERENCE_1_1_REPO:-git@github.com:mlcommons/submissions_inference_1_1.git}"

rm -rf intel_inference_datacenter_v1-1
git clone --single-branch --branch=main ${SUBMISSIONS_INFERENCE_1_1_REPO} intel_inference_datacenter_v1-1
rm -rf intel_inference_datacenter_v1-1/.git
rm -rf intel_inference_datacenter_v1-1/closed/Intel/results
rm -rf intel_inference_datacenter_v1-1/closed/Intel/compliance/

DOCKER_BUILDKIT=1 docker build ${DOCKER_BUILD_ARGS} -f Dockerfile -t ${IMAGE_NAME} .

echo "Building BERT-99 workflow container"

docker run --rm -it ${IMAGE_NAME} python -c "import torch; print('torch:', torch.__version__)"

# docker push intel/intel-optimized-pytorch:v1.7.0-ipex-v1.2.0-dlrm
