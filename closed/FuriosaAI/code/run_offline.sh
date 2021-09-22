#!/bin/bash
mkdir -p ../results/warboy_intel/resnet50/Offline/performance/run_1/
mkdir -p ../results/warboy_intel/resnet50/Offline/accuracy/
mkdir -p ../results/warboy_intel/ssd-mobilenet/Offline/performance/run_1/
mkdir -p ../results/warboy_intel/ssd-mobilenet/Offline/accuracy/
mkdir -p ../results/warboy_intel/ssd-resnet34/Offline/performance/run_1/
mkdir -p ../results/warboy_intel/ssd-resnet34/Offline/accuracy/

# resnet50
cd resnet50
mkdir -p build
./run-resnet50.sh --scenario Offline
cp -R build/* ../../results/warboy_intel/resnet50/Offline/performance/run_1/

./run-resnet50.sh --accuracy --scenario Offline
./run-accuracy-resnet50.sh > build/accuracy.txt
cp -R build/* ../../results/warboy_intel/resnet50/Offline/accuracy/
cd -

# ssd-mobilenet
cd ssd-mobilenet
mkdir -p build

./run-ssd-mobilenet.sh --scenario Offline
cp -R build/* ../../results/warboy_intel/ssd-mobilenet/Offline/performance/run_1/

./run-ssd-mobilenet.sh --accuracy --scenario Offline
./run-accuracy-ssd-mobilenet.sh > build/accuracy.txt
cp -R build/* ../../results/warboy_intel/ssd-mobilenet/Offline/accuracy/
cd -

# ssd-resnet34
cd ssd-resnet34
mkdir -p build

./run-ssd-resnet34.sh --scenario Offline
cp -R build/* ../../results/warboy_intel/ssd-resnet34/Offline/performance/run_1/

./run-ssd-resnet34.sh --accuracy --scenario Offline
./run-accuracy-ssd-resnet34.sh > build/accuracy.txt
cp -R build/* ../../results/warboy_intel/ssd-resnet34/Offline/accuracy/
cd -
