#!/bin/bash
mkdir -p ../results/warboy_intel/resnet50/SingleStream/performance/run_1/
mkdir -p ../results/warboy_intel/resnet50/SingleStream/accuracy/

# resnet50
cd resnet50
mkdir -p build
./run-resnet50.sh
cp -R build/* ../../results/warboy_intel/resnet50/SingleStream/performance/run_1/

./run-resnet50.sh --accuracy
./run-accuracy-resnet50.sh > build/accuracy.txt
cp -R build/* ../../results/warboy_intel/resnet50/SingleStream/accuracy/
cd -
