echo "Test start time: $(date)"
./ov_mlperf --scenario Offline \
	--mlperf_conf Configs/mlperf.conf \
	--user_conf Configs/ssd-resnet34/user.conf \
	--model_name ssd-resnet34 \
	--data_path /home/dell/CK-TOOLS/dataset-coco-2017-val \
  --device CPU \ 
	--batch_size 4 \  
	--nireq 14 \
	--nthreads 56 \
	--nstreams 14 \
	--total_sample_count 1024 \
	--warmup_iters 500 \
	--model_path Models/ssd-resnet34/ssd-resnet34_int8.xml
echo "Test end time: $(date)"

