cd acc_scripts

vocab_file=$1
val_data=$2

python accuracy-squad.py --vocab_file $1 \
--val_data $2 \
--log_file ../test_log/mlperf_log_accuracy.json \
--out_file predictions.json \
2>&1 | tee ../test_log/accuracy.txt
