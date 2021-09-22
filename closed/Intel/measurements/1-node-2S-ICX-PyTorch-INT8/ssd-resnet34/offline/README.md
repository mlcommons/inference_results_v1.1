# Setup from Source

### Install Dependencies

#### Create Anaconda environment
```
sudo apt install g++
wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
chmod +x Anaconda3-2020.02-Linux-x86_64.sh
bash ./Anaconda3-2020.02-Linux-x86_64.sh
conda create --name ssd-rn34_env pytorch=3.7
conda activate ssd-rn34_env
```

#### Install Python and IPex
```
  CUR_DIR=$(pwd)
  git clone <path/to/this/repo>
  ln -s <path/to/this/repo>/closed/Intel/code/ssd-resnet34/pytorch-cpu ssd_pytorch
  cp ssd_pytorch/prepare_env.sh .
  bash prepare_env.sh

```

### Download Model
```
CUR_DIR=$(pwd)
export WORKLOAD_DATA=${CUR_DIR}/data
mkdir -p ${WORKLOAD_DATA}

wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=13kWgEItsoxbVKUlkQz4ntjl1IZGk6_5Z'  -O 'ssd-resnet34.pth'
mv 'ssd-resnet34.pth' ${WORKLOAD_DATA}/

```

### Download Dataset
```
CUR_DIR=$(pwd)
mkdir -p ${CUR_DIR}/dataset-coco
cd ${CUR_DIR}/dataset-coco
curl -O http://images.cocodataset.org/zips/train2017.zip; unzip train2017.zip # If you need to do calibration
curl -O http://images.cocodataset.org/zips/val2017.zip; unzip val2017.zip
curl -O http://images.cocodataset.org/annotations/annotations_trainval2017.zip; unzip annotations_trainval2017.zip
cd ${CUR_DIR}
```

### Update Config file
Update relevant entries of ```config.json```

### Run Benchmark

Navigate to [Root](../../) for detailed instructions
```
conda activate ssd-rn34_env
bash ssd-resent34/pytorch-cpu/scripts/ICX/mlperf_run_offline.sh
 
bash ssd-resent34/pytorch-cpu/scripts/ICX/mlperf_run_server.sh
 
To verify accuracy, edit verify_accuracy.sh to include appropriate paths.
./verify_accuracy.sh
 
```

# Setup with Docker

## Steps to run SSD-Resnet34

### 1.prepare dataset and model in host
The dataset and model of each workload need to be prepared in advance in the host

#### model:

```
export WORKLOAD_DATA=ssd-resnet34/data
mkdir -p ${WORKLOAD_DATA}

wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=13kWgEItsoxbVKUlkQz4ntjl1IZGk6_5Z'  -O 'ssd-resnet34.pth'
mv 'ssd-resnet34.pth' ${WORKLOAD_DATA}/
```

#### dataset:

```
CUR_DIR=$(pwd)
mkdir -p ssd-resnet34/dataset-coco
cd $ssd-resnet34/dataset-coco
curl -O http://images.cocodataset.org/zips/train2017.zip; unzip train2017.zip # If you need to do calibration
curl -O http://images.cocodataset.org/zips/val2017.zip; unzip val2017.zip
curl -O http://images.cocodataset.org/annotations/annotations_trainval2017.zip; unzip annotations_trainval2017.zip
cd ${CUR_DIR}
```

### 2.Start a container

#### 1. run a container

```
docker run --privileged --name intel_ssd_resnet34 -itd --net=host --ipc=host -v <path/to/dataset/and/model>:/root/mlperf_data/ssd-resnet34 intel/intel-optimized-pytorch:v1.8.0-ipex-v1.8.0-ssd-resnet34
# check container, it will show a container named inetl_ssd_resnet34
docker ps
```

#### 2. into container

```
docker exec -it intel_ssd_resnet34 bash
```

### 3.Run SSD-Resnet34

Example run scripts (to run 20 instances on ICX 2 socket 40 core system) in "scripts/ICX" directory.

```
#please update relevant entries of config.json
cd opt/workdir/intel_inference_datacenter_v1-1/closed/Intel/code
#for Offline scenario
bash ssd-resent34/pytorch-cpu/scripts/ICX/mlperf_run_offline.sh

#for Server scenario
bash ssd-resent34/pytorch-cpu/scripts/ICX/mlperf_run_server.sh

# Run Offline accuracy 
bash ssd-resent34/pytorch-cpu/scripts/ICX/mlperf_run_offline_accuracy.sh

# Run Server accuracy
bash ssd-resent34/pytorch-cpu/scripts/ICX/mlperf_run_server_accuracy.sh

#get accuracy result
cd ../..
#To verify accuracy, edit verify_accuracy.sh to include appropriate paths.
bash verify_accuracy.sh
```
