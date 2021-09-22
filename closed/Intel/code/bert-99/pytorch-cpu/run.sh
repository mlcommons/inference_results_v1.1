set -x
export LD_PRELOAD=${CONDA_PREFIX}/lib/libjemalloc.so
export MALLOC_CONF="oversize_threshold:1,background_thread:true,percpu_arena:percpu,metadata_thp:always,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000";

accuracy=$1

sut_dir=$(pwd)
executable=${sut_dir}/build/bert_inference
mode="Offline"
OUTDIR="$sut_dir/test_log"
mkdir $OUTDIR
CONFIG="-n 20 -j 4 --test_scenario=${mode} --model_file=${sut_dir}/bert.pt --sample_file=${sut_dir}/squad.pt --mlperf_config=${sut_dir}/inference/mlperf.conf --user_config=${sut_dir}/user.conf -o ${OUTDIR} -b 64 ${accuracy}"

${executable} ${CONFIG}

if [ ${accuracy} = "--accuracy" ]; then
        vocab_file= #path/to/vocab.txt
        val_data= #path/to/dev-v1.1.json
	bash acc_scripts/accuracy.sh $vocab_file $val_data 
fi

set +x

