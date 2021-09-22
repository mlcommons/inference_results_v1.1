#! /bin/bash

# Create new env and activate it
conda install python=3.7
conda install ninja
conda install cmake
conda install -c intel mkl
conda install -c intel mkl-include
conda install -c intel intel-openmp
conda install -c conda-forge llvm-openmp
conda install jemalloc
