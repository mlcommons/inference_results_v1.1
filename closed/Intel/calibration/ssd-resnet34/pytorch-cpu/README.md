# Setup Instructions

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
mv 'resnet34-ssd1200.pth' ${WORKLOAD_DATA}/

```

### Download Dataset
```
CUR_DIR=$(pwd)
mkdir -p ${CUR_DIR}/dataset-coco
cd ${CUR_DIR}/dataset-coco
curl -O http://images.cocodataset.org/zips/train2017.zip; unzip train2017.zip # If you need to do calibration
cd ${CUR_DIR}
```


### Run Calibration

```
conda activate ssd-rn34_env
export DATA_DIR=${CUR_DIR}/dataset-coco/train2017 
export MODEL_DIR=${WORKLOAD_DATA}/resnet34-ssd1200.pth
bash run_calibration.sh


To verify accuracy, edit verify_accuracy.sh to include appropriate paths.
bash verify_accuracy.sh

```
