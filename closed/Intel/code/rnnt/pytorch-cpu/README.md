# Setup from source

## SW requirements
###
| SW |configuration |
|--|--|
| GCC | GCC 9.3 |

## Steps to run RNNT

### 1. Install anaconda 3.0
```
  wget https://repo.continuum.io/archive/Anaconda3-5.0.0-Linux-x86_64.sh -O anaconda3.sh
  chmod +x anaconda3.sh
  ~/anaconda3.sh -b -p ~/anaconda3
  ~/anaconda3/bin/conda create -n rnnt python=3.7

  export PATH=~/anaconda3/bin:$PATH
  source ~/anaconda3/bin/activate rnnt
```
### 2. Prepare code and enviroment
```
  git clone <path/to/this/repo>
  cd <path/to/this/repo>/closed/Intel/code/rnnt
  bash prepare_env.sh
```
### 3. Prepare model and dataset
```
  work_dir=mlperf-rnnt-librispeech
  local_data_dir=$work_dir/local_data
  mkdir -p $local_data_dir
  librispeech_download_dir=.
  # prepare model
  wget https://zenodo.org/record/3662521/files/DistributedDataParallel_1576581068.9962234-epoch-100.pt?download=1 -O $work_dir/rnnt.pt

  # prepare inference dataset
  wget https://www.openslr.org/resources/12/dev-clean.tar.gz
  # suggest you check run.sh to locate the dataset
  python pytorch/utils/download_librispeech.py \
         pytorch/utils/librispeech-inference.csv \
         $librispeech_download_dir \
         -e $local_data_dir --skip_download
  python pytorch/utils/convert_librispeech.py \
         --input_dir $local_data_dir/LibriSpeech/dev-clean \
         --dest_dir $local_data_dir/dev-clean-wav \
         --output_json $local_data_dir/dev-clean-wav.json
```
### 4. Calibration
```
  # prepare calibration dataset and file list
  wget https://www.openslr.org/resources/12/train-clean-100.tar.gz
  wget https://raw.githubusercontent.com/mlcommons/inference/master/calibration/LibriSpeech/calibration_files.txt
  tar zxvf train-clean-100.tar.gz -C $local_data_dir
  python pytorch/utils/convert_librispeech.py \
         --input_dir $local_data_dir/LibriSpeech/train-clean-100 \
         --dest_dir $local_data_dir/train-clean-100-wav \
         --dest_list calibration_files.txt \
         --output_json $local_data_dir/train-clean-100-wav.json

  # calibration
  ../../../calibration/rnnt/pytorch-cpu/calibration.sh
  # this will generate a calibration_result.json
```
### 5. Run RNN-T
  Please update the setup_env_offline.sh or setup_evn_server.sh and user.conf according to your platform resource.
```
  export TCMALLOC_DIR=$CONDA_PREFIX/lib
  # offline
  sudo ./run_clean.sh
  source ./setup_env_offline.sh
  ./run_inference_cpu.sh
  # offline accuracy
  sudo ./run_clean.sh
  source ./setup_env_offline.sh
  ./run_inference_cpu.sh --accuracy
  # server scenario
  sudo ./run_clean.sh
  source ./setup_env_server.sh
  ./run_inference_cpu.sh --server
  # server accuracy
  sudo ./run_clean.sh
  source ./setup_env_server.sh
  ./run_inference_cpu.sh --accuracy
```
### Note on Server scenario
```
For server scenario, we exploit the fact that incoming data have different sequence lengths (and inference times) by bucketing according to sequence length 
and specifying batch size for each bucket such that latency can be satisfied. The settings are specified in machine.conf file and required fields 
are cores_per_instance, num_instances, waveform_len_cutoff, batch_size.
```
# Setup with Docker

## Steps to run RNNT

### 1.Prepare dataset and model in host

#The dataset and model of each workload need to be prepared in advance in the host

```
work_dir=rnnt/mlperf-rnnt-librispeech
local_data_dir=$work_dir/local_data
mkdir -p $local_data_dir
cd $local_data_dir
wget https://www.openslr.org/resources/12/dev-clean.tar.gz
wget https://zenodo.org/record/3662521/files/DistributedDataParallel_1576581068.9962234-epoch-100.pt?download=1 -O $work_dir/rnnt.pt
```

#prepare calibration dataset

```
wget https://www.openslr.org/resources/12/train-clean-100.tar.gz
wget https://raw.githubusercontent.com/mlcommons/inference/master/calibration/LibriSpeech/calibration_files.txt
mv train-clean-100.tar.gz $local_data_dir
cd $local_data_dir
tar zxvf train-clean-100.tar.gz
```

### 2.Start a container

#start a container

```
#-v mount the dataset to docker container
docker run --privileged --name intel_rnnt -itd --net=host --ipc=host -v </path/to/dataset/and/model>:/root/mlperf_data/rnnt intel/intel-optimized-pytorch:v1.8.0-ipex-v1.8.0-rnnt
#check container, it will show a container named intel_rnnt
docker ps
```

#into container

```
docker exec -it intel_rnnt bash
```

### 3.Dataset convert and calibration

```
#dataset convert
cd /opt/workdir/intel_mlperf_inference/closed/Intel/code/rnnt/pytorch-cpu
ln -s /root/mlperf_data/rnnt/mlperf-rnnt-librispeech mlperf-rnnt-librispeech
work_dir=mlperf-rnnt-librispeech
local_data_dir=$work_dir/local_data
librispeech_download_dir=$local_data_dir

conda install pandas
#This will hae a non-zero exit code if the checksum is incorrect
python pytorch/utils/download_librispeech.py \
pytorch/utils/librispeech-inference.csv \
$librispeech_download_dir \
-e $local_data_dir --skip_download

#We need to convert .flac files to .wav files via sox.
export PATH=/opt/workdir/third_party/local/bin:$PATH
python pytorch/utils/convert_librispeech.py \
--input_dir mlperf-rnnt-librispeech/local_data/LibriSpeech/dev-clean \
--dest_dir mlperf-rnnt-librispeech/local_data/dev-clean-wav \
--output_json mlperf-rnnt-librispeech/local_data/dev-clean-wav.json

#prepare calibration dataset
python pytorch/utils/convert_librispeech.py \
--input_dir mlperf-rnnt-librispeech/local_data/LibriSpeech/train-clean-100 \
--dest_dir mlperf-rnnt-librispeech/local_data/train-clean-100-wav \
--dest_list mlperf-rnnt-librispeech/local_data/calibration_files.txt \
--output_json mlperf-rnnt-librispeech/local_data/train-clean-100-wav.json

#calibration
export LD_LIBRARY_PATH=/opt/conda/lib/python3.7/site-packages/torch_ipex-1.8.0-py3.7-linux-x86_64.egg/lib:$LD_LIBRARY_PATH
./calibration.sh
#this will generate a calibration_result.json
```

### 4.Run RNN-T

#Please update the setup_env_offline.sh or setup_env_server.sh and user.conf according to your platform resource.

```
export TCMALLOC_DIR=$CONDA_PREFIX/lib
```

#offline

```
sudo ./run_clean.sh
source ./setup_env_offline.sh
./run_inference_cpu.sh
```

#offline accuracy

```
sudo ./run_clean.sh
source ./setup_env_offline.sh
./run_inference_cpu.sh --accuracy
```

#server

```
sudo ./run_clean.sh
source ./setup_env_server.sh
./run_inference_cpu.sh --server
```

#server accuracy

```
sudo ./run_clean.sh
source ./setup_env_server.sh
./run_inference_cpu.sh --accuracy --server
```

note:
For server scenario, we exploit the fact that incoming data have different sequence lengths (and inference times) by bucketing according to sequence length and specifying batch size for each bucket such that latency can be satisfied. The settings are specified in server.conf file and required fields are cores_per_instance, num_instances, waveform_len_cutoff, batch_size.
