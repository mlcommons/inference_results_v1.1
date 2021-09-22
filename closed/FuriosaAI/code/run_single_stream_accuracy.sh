#!/bin/bash
# resnet50
cd resnet50
mkdir -p build
./run-resnet50.sh --accuracy
./run-accuracy-resnet50.sh
cd -

# ssd-mobilenet
cd ssd-mobilenet
mkdir -p build
./run-ssd-mobilenet.sh --accuracy
./run-accuracy-ssd-mobilenet.sh
cd -

# ssd-resnet34
cd ssd-resnet34
mkdir -p build
./run-ssd-resnet34.sh --accuracy
./run-accuracy-ssd-resnet34.sh
cd -
