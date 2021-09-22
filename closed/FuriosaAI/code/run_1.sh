#!/bin/bash
mkdir -p ../results/warboy_intel/resnet50/Offline/performance/run_1/
mkdir -p ../results/warboy_intel/resnet50/Offline/accuracy/

# resnet50
cd resnet50
mkdir -p build
./run-resnet50.sh --scenario Offline
cp -R build/* ../../results/warboy_intel/resnet50/Offline/performance/run_1/

./run-resnet50.sh --accuracy --scenario Offline
./run-accuracy-resnet50.sh > build/accuracy.txt
cp -R build/* ../../results/warboy_intel/resnet50/Offline/accuracy/
cd -
