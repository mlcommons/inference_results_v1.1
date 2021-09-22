# Setup from Source

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
  ln -s <path/to/this/repo>/closed/Intel/code/dlrm/pytorch-cpu dlrm_pytorch
  cp dlrm_pytorch/prepare_conda_env.sh .
  bash ./prepare_conda_env.sh
  cp dlrm_pytorch/prepare_env.sh .
  bash prepare_env.sh
```
### 3. Prepare DLRM dataset and code    
(1) Prepare DLRM dataset
```
   Create a directory (such as ${WORKDIR}\dataset\terabyte_input) which contain:
     day_fea_count.npz
     terabyte_processed_test.bin

   About how to get the dataset, please refer to
      https://github.com/facebookresearch/dlrm
```
(2) Prepare pre-trained DLRM model
```
   cd dlrm_pytorch/python/model
   wget https://dlrm.s3-us-west-1.amazonaws.com/models/tb00_40M.pt -O dlrm_terabyte.pytorch
```
### 4. Run command for server and offline mode

(1) cd dlrm_pytorch

(2) configure DATA_DIR and MODEL_DIR #you can modify the setup_dataset.sh, then 'source ./setup_dataset.sh'
```
   export DATA_DIR=           # the path of dataset, for example as ${WORKDIR}\dataset\terabyte_input
   export MODEL_DIR=          # the path of pre-trained model, for example as ${WORKDIR}\dlrm_pytorch\python\model
```
(3) configure offline/server mode options # currenlty used options for each mode is in setup_env_xx.sh, You can modify it, then 'source ./setup_env_xx.sh'
```
   export NUM_SOCKETS=        # i.e. 8
   export CPUS_PER_SOCKET=    # i.e. 28
   export CPUS_PER_PROCESS=   # i.e. 14. which determine how many cores for one processe running on one socket
                              #   process_number = $CPUS_PER_SOCKET / $CPUS_PER_PROCESS
   export CPUS_PER_INSTANCE=  # i.e. 14. which determine how many cores used for one instance inside one process
                              #   instance_number_per_process = $CPUS_PER_PROCESS / CPUS_PER_INSTANCE
                              #   total_instance_number_in_system = instance_number_per_process * process_number
```
(4) command line
   Please updae setup_env_server.sh and setup_env_offline.sh and user.conf according to your platform resource.
```
   # server-performance-mode
   sudo ./run_clean.sh
   source ./setup_env_server.sh
   ./run_main.sh server           #run for DLRM fp32 model
   ./run_main.sh server int8      #run for DLRM int8 model
   ./run_main.sh server bf16      #run for DLRM bf16 model

   # server-accuracy-mode
   sudo ./run_clean.sh
   source ./setup_env_server.sh
   ./run_main.sh server accuracy           #run for DLRM fp32 model
   ./run_main.sh server accuracy int8      #run for DLRM int8 model
   ./run_main.sh server accuracy bf16      #run for DLRM bf16 model

   # offline-performance-mode
   sudo ./run_clean.sh
   source ./setup_env_offline.sh
   ./run_main.sh offline           #run for DLRM fp32 model
   ./run_main.sh offline int8      #run for DLRM int8 model
   ./run_main.sh offline bf16      #run for DLRM bf16 model

   # offline-accuracy-mode
   sudo ./run_clean.sh
   source ./setup_env_offline.sh
   ./run_main.sh offline accuracy           #run for DLRM fp32 model
   ./run_main.sh offline accuracy int8      #run for DLRM int8 model
   ./run_main.sh offline accuracy bf16      #run for DLRM bf16 model

   for int8 calibration scripts, please look into Intel/calibration/dlrm/pytorch-cpu/ directory.
   for int8 execution, calibration result is in int8_configure.json which is copied from that output.
```

# Setup with Docker

## Steps to run DLRM

### 1.Prepare dataset and model in host

The dataset and model of each workload need to be prepared in advance in the host

#### dataset：

about how to get the dataset please refer to https://github.com/mlcommons/inference/tree/master/recommendation/dlrm/pytorch.
create a directory (such as dlrm\dataset\terabyte_input) which contain day_day_count.npz, day_fea_count.npz and terabyte_processed_test.bin#For calibration, need one of the following two files:terabyte_processed_val.bin or calibration.npz

#### model:

```
mkdir dlrm/dlrm_pytorch/python/model
cd dlrm/dlrm_pytorch/python/model
wget https://dlrm.s3-us-west-1.amazonaws.com/models/tb00_40M.pt -O dlrm_terabyte.pytorch
```

### 2.Start and log into a container

#### (1) start a container

-v mount the dataset to docker container 

```
docker run --privileged --name intel_dlrm -itd --net=host --ipc=host -v </path/to/dataset/and/model>:/root/mlperf_data/dlrm intel/intel-optimized-pytorch:v1.7.0-ipex-v1.2.0-dlrm
```

check container, it will show a container named intel_dlrm

```
docker ps
```

#### (2) into container

```
docker exec -it mlperf_dlrm bash
```

### 3.Run DLRM

#### (1) cd dlrm_pytorch

```
cd /opt/workdir/intel_inference_datacenter_v1-1/closed/Intel/code/dlrm/pytorch-cpu
```

#### (2) configure DATA_DIR and MODEL_DIR 

you can modify the setup_dataset.sh, then 'source ./setup_dataset.sh'

```
export DATA_DIR=    # the path of dataset, for example as ${WORKDIR}\dataset\terabyte_input
export MODEL_DIR=    # the path of pre-trained model, for example as ${WORKDIR}\dlrm_pytorch\python\model
```

#### (3) configure offline/server mode options 

currenlty used options for each mode is in setup_env_offline/server.sh, You can modify it, then 'source ./setup_env_offline/server.sh'

```
export NUM_SOCKETS=        # i.e. 8
export CPUS_PER_SOCKET=    # i.e. 28
export CPUS_PER_PROCESS=   # i.e. 14. which determine how many cores for one processe running on one socket
                           #   process_number = $CPUS_PER_SOCKET / $CPUS_PER_PROCESS
export CPUS_PER_INSTANCE=  # i.e. 14. which determine how many cores used for one instance inside one process
                           #   instance_number_per_process = $CPUS_PER_PROCESS / CPUS_PER_INSTANCE
                           #   total_instance_number_in_system = instance_number_per_process * process_number
```

#### (4) command line

Please update setup_env_server.sh and setup_env_offline.sh and user.conf according to your platform resource.

```
bash run_mlperf.sh --mode=<offline/server> --type=<perf/acc> --dtype=int8
```

for int8 calibration scripts, please look into Intel/calibration/dlrm/pytorch-cpu/ directory.

for int8 execution, calibration result is in int8_configure.json which is copied from that output.
