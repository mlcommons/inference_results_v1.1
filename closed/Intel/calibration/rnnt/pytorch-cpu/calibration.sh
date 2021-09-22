CODE_DIR=${PWD}  # <repo-path>/closed/Intel/calibration/rnnt/pytorch-cpu
CALIBRATION_DIR=`dirname "$0"`  # <repo-path>/closed/Intel/code/rnnt/pytorch-cpu
export PYTHONPATH="${CODE_DIR}:${CODE_DIR}/pytorch:${PYTHONPATH}"

python ${CALIBRATION_DIR}/calibration.py \
    --manifest ${CODE_DIR}/mlperf-rnnt-librispeech/local_data/train-clean-100-wav.json \
    --dataset_dir ${CODE_DIR}/mlperf-rnnt-librispeech/local_data \
    --configure_save_path ${CODE_DIR}/calibration_result.json \
    --pytorch_checkpoint ${CODE_DIR}/mlperf-rnnt-librispeech/rnnt.pt
