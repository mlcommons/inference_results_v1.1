# Setup from Source
## Please follow the steps to build env
```
  conda create -n bert-pt
  source activate bert-pt

  bash prepare_conda.sh


 # please add the dataset and model path 
  bash prepare_env.sh
```

Tester provide mlperf.conf and user.conf
See ```prepare_env.sh``` and ```run.sh```


## dataset and model

 dataset

```
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -O ./squad/dev-v1.1.json
```
 model

```
git clone https://huggingface.co/bert-large-uncased
 #replace  pytorch_model.bin
wget https://zenodo.org/record/4792496/files/pytorch_model.bin?download=1 -O pytorch_model.bin
```


## Run command

 #for accuracy mode, you should install tensorflow
 #please update the vocab.txt and dev-v1.1.json path in run.sh and run_server.sh.
```
bash run.sh    #for offline performance
bash run.sh --accuracy   #for offline accuracy
```
```
bash run_server.sh #for server performance
bash run_server.sh --accuracy    #for server accuracy
```
inference driver options:
    ("-m, --model_file", "[filename] Torch Model File")

    ("-s, --sample_file", "[filename] SQuAD Sample File")

    ("-t, --test_mode", "Test mode [Offline, Server]")

    ("-n, --inter_parallel", "[number] Instance Number")

    ("-j, --intra_parallel", "[number] Thread Number Per-Instance")

    ("-c, --mlperf_config", "[filename] Configuration File for LoadGen")

    ("-u, --user_config", "[filename] User Configuration for LoadGen")

    ("-o, --output_dir", "[filename] Test Output Directory")

    ("-b, --batch", "[number] Offline Model Batch Size")

    ("-h, --hyperthreading", "[true/false] Whether system enabled hyper-threading or not")


For ICX above, subsitute: -mavx512cd -mavx512dq -mavx512bw -mavx512vl to -march=native


# Setup with docker
## prepare dataset and model
 follow the steps above

## start docker container
```
docker run --privileged --name intel_bert -itd --net=host --ipc=host -v </path/to/datatset/and/model>:/root/mlperf_date/bert intel/intel-optimized-pytorch:v1.9.0-bert
```

## convert dataset and model
```
cd /opt/workdir/intel_inference_datacenter_v1-1/closed/Intel/code/bert-99/pytorch-cpu/
cd model
python save_bert_inference.py -m $model_dir -o ../bert.pt
cd ../datasets
python save_squad_features.py -m $model_dir -d $dataset_dir -o ../squad.pt
cd ..
```
## Run command
follow the steps above









