# DLRM MLPerf Inference v1.0 Intel Submission

## HW and SW requirements
### 1. HW requirements
| HW | configuration |
| -: | :- |
| CPU | ICX-6 @ 2 sockets/Node |
| DDR | 512G/socket @ 3200 MT/s |
| SSD | 1 SSD/Node @ >= 1T |

### 2. SW requirements
| SW |configuration  |
|--|--|
| GCC | GCC 9.3  |

## Steps to run DLRM

### 1. Install anaconda 3.0
```
  wget -c https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh -O anaconda3.sh
  chmod +x anaconda3.sh
  ~/anaconda3.sh -b -p ~/anaconda3
  ~/anaconda3/bin/conda create -n dlrm python=3.7
  export PATH=/usr/bin:~/anaconda3/bin:$PATH
  source ~/anaconda3/bin/activate dlrm
```
### 2. Install dependency packages and Pytorch/IPEX
```
  mkdir <workfolder>
  cd <workfolder>
  git clone <path/to/this/repo>
  ln -s <path/to/this/repo>/closed/Intel/calibration/dlrm/pytorch-cpu dlrm_pytorch_calib
  cp dlrm_pytorch_calib/prepare_conda_env.sh .
  bash ./prepare_conda_env.sh
  cp dlrm_pytorch/prepare_env.sh .
  bash prepare_env.sh
```
### 3. Prepare DLRM dataset and code    
(1) Prepare Calibration dataset
```
   For calibration, need the following two files:
     day_fea_count.npz
     terabyte_processed_val.bin
   
   For DLRM MLPerf Int8 Inference, we use the first 128000 rows (user-item pairs) of the second half of day_23 as the calibration set.
   terabyte_processed_val.bin is the second part of day_23 which bin rows is start from the 89137319-th row of day_23.

   About how to get the day_fea_cout.npz and terabyte_processed_val.bin, please refer to
      https://github.com/facebookresearch/dlrm
```
(2) Prepare pre-trained DLRM model
```
   cd dataset 
   wget https://dlrm.s3-us-west-1.amazonaws.com/models/tb00_40M.pt -O dlrm_terabyte.pytorch
```
### 4. Run command to do calibration 
(1) cd dlrm_pytorch_calib
(2) configure DATA_DIR and MODEL_DIR #you can modify the setup_dataset.sh, then 'source ./setup_dataset.sh'
(3) command line
```
   # do calibration
   bash ./run_calibrate.sh # run for int8 calibration
   #calibration output is int8_configure.json under output/
   cp ./output/int8_configure.json to dlrm pytorch directory for accuracy and performance testing
```
