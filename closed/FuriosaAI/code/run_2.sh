#!/bin/bash
mkdir -p ../results/warboy_intel/ssd-mobilenet/SingleStream/performance/run_1/
mkdir -p ../results/warboy_intel/ssd-mobilenet/SingleStream/accuracy/
mkdir -p ../results/warboy_intel/ssd-mobilenet/Offline/performance/run_1/
mkdir -p ../results/warboy_intel/ssd-mobilenet/Offline/accuracy/

# ssd-mobilenet
cd ssd-mobilenet
mkdir -p build

./run-ssd-mobilenet.sh
cp -R build/* ../../results/warboy_intel/ssd-mobilenet/SingleStream/performance/run_1/

./run-ssd-mobilenet.sh --accuracy
./run-accuracy-ssd-mobilenet.sh > build/accuracy.txt
cp -R build/* ../../results/warboy_intel/ssd-mobilenet/SingleStream/accuracy/
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

