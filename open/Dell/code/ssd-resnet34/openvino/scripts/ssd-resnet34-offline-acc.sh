echo "Test start time: $(date)"
./Release/ov_mlperf --scenario Offline \
  --mode Accuracy \
	--mlperf_conf Configs/mlperf.conf \
	--user_conf Configs/ssd-resnet34/user.conf \
	--model_name ssd-resnet34 \
	--data_path /home/dell/CK-TOOLS/dataset-coco-2017-val \
  --device CPU \
	--nireq 14 \
	--nthreads 56 \
	--nstreams 14 \
	--total_sample_count 5000 \
	--warmup_iters 50 \
	--model_path Models/ssd-resnet34/ssd-resnet34_int8.xml
echo "Test end time: $(date)"