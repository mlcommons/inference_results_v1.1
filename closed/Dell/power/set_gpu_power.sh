#!/bin/bash

PL=175

sudo nvidia-smi -pm 0  
sleep 1
sudo nvidia-smi -pm 1 
sleep 1
sudo nvidia-smi -rgc 
sleep 1
sudo nvidia-smi -pl ${PL} 
sleep 1
nvidia-smi
