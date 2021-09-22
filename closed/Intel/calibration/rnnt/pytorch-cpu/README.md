# RNNT MLPerf Calibration BKC

## SW requirements
###
| SW |configuration |
|--|--|
| GCC | GCC 9.3 |

## Steps to do calibration for RNNT

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
### 3. Prepare model
```
  work_dir=mlperf-rnnt-librispeech
  # prepare model
  wget https://zenodo.org/record/3662521/files/DistributedDataParallel_1576581068.9962234-epoch-100.pt?download=1 -O $work_dir/rnnt.pt
```
### 4. Calibration
```
  local_data_dir=$work_dir/local_data
  mkdir -p $local_data_dir
  librispeech_download_dir=.
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