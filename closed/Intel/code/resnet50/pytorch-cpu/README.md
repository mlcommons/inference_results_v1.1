# Setup from Source

### Install Dependencies

#### Create Anaconda environment
```
sudo apt install g++
wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
chmod +x Anaconda3-2020.02-Linux-x86_64.sh
bash ./Anaconda3-2020.02-Linux-x86_64.sh
conda create -n resnet50_pt_env python=3.7
source activate resnet50_pt_env
```

#### Install Pytorch and Ipex 
```
CUR_DIR=$(pwd)
bash ../../prepare_pytorch.sh $CUR_DIR
```

#### Install MLPerf Loadgen

```
bash ../../prepare_loadgen.sh $CUR_DIR
```
#### Install Dependencies for Resnet50

```
cd resnet50/pytorch-cpu/
./prepare_env.sh --code=<this repo>
```

### Download Model
```
CUR_DIR=$(pwd)
export WORKLOAD_DATA=${CUR_DIR}/data
mkdir -p ${WORKLOAD_DATA}

wget --no-check-certificate 'https://zenodo.org/record/4588417/files/resnet50-19c8e357.pth' -O 'resnet50.pth'
mv 'resnet50.pth' ${WORKLOAD_DATA}/

```

### Download Imagenet Dataset
Instructions at https://image-net.org/download.php.

### Update Config file
Update relevant entries of ```resnet50-config_*.yml```

### Run Benchmark


Example run scripts (to run 20 instances on ICX 2 socket 40 core system)
in "scripts/ICX" directory.
```

cd closed/Intel/code 
#for Offline scenario
bash resnet50/pytorch-cpu/scripts/ICX/mlperf_run_offline.sh
bash resnet50/pytorch-cpu/scripts/ICX/mlperf_run_offline_acc.sh

#for Server scenario
bash resnet50/pytorch-cpu/scripts/ICX/mlperf_run_server.sh
bash resnet50/pytorch-cpu/scripts/ICX/mlperf_run_server_acc.sh

#get accuracy result
python accuracy-imagenet.py --mlperf-accuracy-file mlperf_log_accuracy.json --imagenet-val-file <ILSVRC2012_img_val>/val_map.txt --dtype int32

# Run Offline accuracy 
./mlperf_run_offline_accuracy.sh

# Run Server scenario
./mlperf_run_server_accuracy.sh
```

# Setup with Docker

## Steps to run Resnet50

### 1.prepare dataset and model in host

 #The dataset and model of each workload need to be prepared in advance in the host

#### model:

```
export WORKLOAD_DATA=resnet50/data
mkdir -p ${WORKLOAD_DATA}

wget --no-check-certificate 'https://zenodo.org/record/4588417/files/resnet50-19c8e357.pth' -O 'resnet50.pth'
mv 'resnet50.pth' ${WORKLOAD_DATA}/
```

#### dataset:

Instructions at https://image-net.org/download.php.

### 2.load images

```
docker pull intel/intel-optimized-pytorch:v1.8.0-ipex-v1.8.0-resent50
# check images, it will show a image named intel/intel-optimized-pytorch:v1.8.0-ipex-v1.8.0-resent50
docker images
```

### 3.Start and log in a container

#### (1) start a container

```
#-v mount the dataset to docker container 
docker run --privileged --name intel_resnet50 -itd --net=host --ipc=host -v <path/to/dataset/and/model>:/root/mlperf_data/resnet50 intel/intel-optimized-pytorch:v1.8.0-ipex-v1.8.0-resent50
# check container, it will show a container named intel_resnet50
docker ps
```

#### (2) into container

```
docker exec -it intel_resnet50 bash
```

### 4.Run Resnet50

Example run scripts (to run 20 instances on ICX 2 socket 40 core system) in "scripts/ICX" directory.
```
# please set the dataset and model path in scripts/ICX/resnet50-config_*.yml
cd opt/workdir/intel_inference_datacenter_v1-1/closed/Intel/code
#for Offline scenario
bash resnet50/pytorch-cpu/scripts/ICX/mlperf_run_offline.sh
bash resnet50/pytorch-cpu/scripts/ICX/mlperf_run_offline_acc.sh

#for Server scenario
bash resnet50/pytorch-cpu/scripts/ICX/mlperf_run_server.sh
bash resnet50/pytorch-cpu/scripts/ICX/mlperf_run_server_acc.sh

#get accuracy result
python accuracy-imagenet.py --mlperf-accuracy-file mlperf_log_accuracy.json --imagenet-val-file <ILSVRC2012_img_val>/val_map.txt --dtype int32

# Run Offline accuracy 
./mlperf_run_offline_accuracy.sh

# Run Server scenario
./mlperf_run_server_accuracy.sh
```
