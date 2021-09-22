#!/bin/bash
mkdir -p ../results/warboy_intel/resnet50/SingleStream/performance/run_1/
mkdir -p ../results/warboy_intel/resnet50/SingleStream/accuracy/
mkdir -p ../results/warboy_intel/ssd-mobilenet/SingleStream/performance/run_1/
mkdir -p ../results/warboy_intel/ssd-mobilenet/SingleStream/accuracy/
mkdir -p ../results/warboy_intel/ssd-resnet34/SingleStream/performance/run_1/
mkdir -p ../results/warboy_intel/ssd-resnet34/SingleStream/accuracy/

# resnet50
cd resnet50
mkdir -p build
./run-resnet50.sh
cp -R build/* ../../results/warboy_intel/resnet50/SingleStream/performance/run_1/

./run-resnet50.sh --accuracy
./run-accuracy-resnet50.sh > build/accuracy.txt
cp -R build/* ../../results/warboy_intel/resnet50/SingleStream/accuracy/
cd -

# ssd-mobilenet
cd ssd-mobilenet
mkdir -p build

./run-ssd-mobilenet.sh
cp -R build/* ../../results/warboy_intel/ssd-mobilenet/SingleStream/performance/run_1/

./run-ssd-mobilenet.sh --accuracy
./run-accuracy-ssd-mobilenet.sh > build/accuracy.txt
cp -R build/* ../../results/warboy_intel/ssd-mobilenet/SingleStream/accuracy/
cd -

# ssd-resnet34
cd ssd-resnet34
mkdir -p build

./run-ssd-resnet34.sh
cp -R build/* ../../results/warboy_intel/ssd-resnet34/SingleStream/performance/run_1/

./run-ssd-resnet34.sh --accuracy
./run-accuracy-ssd-resnet34.sh > build/accuracy.txt
cp -R build/* ../../results/warboy_intel/ssd-resnet34/SingleStream/accuracy/
cd -
