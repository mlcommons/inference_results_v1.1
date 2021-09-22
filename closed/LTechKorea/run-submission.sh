#!/bin/bash

set -e
#set -x

TOP_DIR=$(pwd)
SUBMITTER="LTechKorea"

#TOPOLOGY="3d-unet bert dlrm rnnt resnet50 ssd-resnet34"
TOPOLOGY="3d-unet ssd-resnet34"
#FAST_OPT="--fast"

SCENARIO="Offline Server"
ACCURACY="default"

echo -e "\n\n START!!! ===== $(date) =====\n"

#make prebuild
if [ -e  "${TOP_DIR}/build" ] ; then
  echo "Delete Previous Logs..."
  rm -rf "${TOP_DIR}/build/logs"
else
  echo "Building pre-requisites..."
  make launch_docker DOCKER_COMMAND='make build'
  echo -e "===== $(date) =====\n"
fi

#
#make download_data BENCHMARKS=resnet50
#make launch_docker DOCKER_COMMAND='make preprocess_data BENCHMARKS=resnet50'

for topology in ${TOPOLOGY}
do
  SCENARIO="Offline,Server"
  ACCURACY="default"

  echo -e "\nStart for ${topology}"
  if [ "${topology}" == "3d-unet" ] ; then
    SCENARIO="Offline"
  fi
  if [ "${topology}" == "3d-unet" ] ||[ "${topology}" == "bert" ] \
    || [ "${topology}" == "dlrm" ] ; then
    ACCURACY="default,high_accuracy"
  fi

  RUN_ARGS="--benchmarks=${topology} --scenarios=${scenario} --config_ver=${ACCURACY}"
  RUN_ARGS="${RUN_ARGS} --verbose"

  for scenario in $(echo ${SCENARIO/','/ })
  do

    for accuracy in $(echo ${ACCURACY/','/ })
    do
      RUN_ARGS="--benchmarks=${topology} \
        --scenarios=${scenario} \
        --config_ver=${accuracy} \
        --verbose ${FAST_OPT}"
      RUN_ARGS=$(echo ${RUN_ARGS} | tr -s ' ')

      echo -e "\n\n START!!! ===== $(date) =====\n"
      echo "[${topology}_${scenario}_${accuracy}] Generating Engines..."
      echo -e "===== $(date) =====\n"
      make launch_docker DOCKER_COMMAND="make generate_engines RUN_ARGS=\"${RUN_ARGS}\""

      TEST_MODE="PerformanceOnly,AccuracyOnly"

      for test_mode in $(echo ${TEST_MODE/','/ })
      do
        RUN_ARGS="--benchmarks=${topology} \
          --scenarios=${scenario} \
          --config_ver=${accuracy} \
          --test_mode=${test_mode} \
          --verbose ${FAST_OPT}"
        RUN_ARGS=$(echo ${RUN_ARGS} | tr -s ' ')

        echo "[${topology}_${scenario}_${accuracy}] Running Harness(${test_mode})..."
        echo -e "===== $(date) =====\n"
        make launch_docker DOCKER_COMMAND="make run_harness RUN_ARGS=\"${RUN_ARGS}\""
      done

      echo "[${topology}_${scenario}_${accuracy}] Update Results..."
      echo -e "===== $(date) =====\n"
      make update_results

      echo "[${topology}_${scenario}_${accuracy}] Running Compliance..."
      echo -e "===== $(date) =====\n"
      make launch_docker DOCKER_COMMAND="make run_audit_harness RUN_ARGS=\"${RUN_ARGS}\""
      echo "[${topology}_${scenario}_${accuracy}] Update Compliance..."
      echo -e "===== $(date) =====\n"
      make update_compliance
      echo -e "END!!! ===== $(date) =====\n\n"
    done  # accuracy
  done  # scenario
done  # topology

#exit 0

# Truncate Accuracy Logs
echo "[ ${SUBMITTER} ] Truncate Accuracy Logs..."
echo -e "===== $(date) =====\n"
make truncate_results SUBMITTER=${SUBMITTER}

# Subission Checker
echo "[ ${SUBMITTER} ] Submission Checking..."
echo -e "===== $(date) =====\n"
make check_submission SUBMITTER=${SUBMITTER}

# Encrypting for Submission
echo "[ ${SUBMITTER} ] Encrypting and Packing..."
echo -e "===== $(date) =====\n"
bash scripts/pack_submission.sh --pack
echo -e "===== $(date) =====\n"

