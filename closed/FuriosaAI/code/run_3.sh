#!/bin/bash
mkdir -p ../results/warboy_intel/ssd-resnet34/SingleStream/performance/run_1/
mkdir -p ../results/warboy_intel/ssd-resnet34/SingleStream/accuracy/
mkdir -p ../results/warboy_intel/ssd-resnet34/Offline/performance/run_1/
mkdir -p ../results/warboy_intel/ssd-resnet34/Offline/accuracy/

# ssd-resnet34
cd ssd-resnet34
mkdir -p build

./run-ssd-resnet34.sh
cp -R build/* ../../results/warboy_intel/ssd-resnet34/SingleStream/performance/run_1/

./run-ssd-resnet34.sh --accuracy
./run-accuracy-ssd-resnet34.sh > build/accuracy.txt
cp -R build/* ../../results/warboy_intel/ssd-resnet34/SingleStream/accuracy/
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
