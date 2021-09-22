#! /bin/bash
set -x
home=$(pwd)

# Provide these two
model_dir=
dataset_dir=
conda install -c conda-forge llvm-openmp

git clone https://github.com/pytorch/pytorch.git

pushd pytorch
git checkout 3979cb0656fe2d8b0445768a769bd624b10778b5
pip install -r requirements.txt
git apply $home/patches/clean_openmp.patch
git apply $home/patches/pytorch.patch
USE_CUDA=OFF python -m pip install -e .
popd


git clone https://github.com/huggingface/transformers.git
pushd transformers
git checkout 9f4e0c23d68366985f9f584388874477ad6472d8
git apply ../patches/transformers.patch
python -m pip install -e .
popd

git clone --branch r1.1 --recursive https://github.com/mlcommons/inference.git


pushd mlperf_plugins
git clone https://github.com/oneapi-src/oneDNN.git onednn
cd onednn
git checkout eac1e4a51a2c64129ec6475b8053b06463b667a9
git apply ../../patches/onednn.patch
popd

mkdir build
pushd build
cmake -DCMAKE_PREFIX_PATH="$(dirname $(python3 -c 'import torch; print(torch.__file__)'))" -GNinja -DCMAKE_BUILD_TYPE=Release ..
ninja
popd

pushd models
python save_bert_inference.py -m $model_dir -o $home/bert.pt
popd

pip install tokenization
pushd datasets
python save_squad_features.py -m $model_dir -d $dataset_dir -o $home/squad.pt
popd
