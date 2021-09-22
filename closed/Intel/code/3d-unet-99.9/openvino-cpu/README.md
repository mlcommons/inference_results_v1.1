#  OpenVINO Int8 Workflow In a Nutshell

To run OpenVino backend please install first [OpenVino 2021.4] (https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/choose-download.html)

To run OpenVino inference benchmarks you would need to convert ONNX model into OpenVino IR format. Int8 inference will require additional step, for that you would need to quantize model using calibrations data. Please refer to the calibration section for instructions on the model conversion to IR as well as the calibration process.


# Loadgen installation

MLPerf loadgen need to be installed before running model benchmarks or model calibration.

1. git clone https://github.com/mlperf/inference.git --depth 1
2. pip install pybind11
3. cd loadgen; 
4. mkdir build
5. cd build
6. cmake ..
7. make

# Model Calibration

To run Int8 inference you would need to calibrate model using calibration data and calibration script (Please refer to the calibration section for instructions on this).

# Building OpenVino C++ SUT

1. cd cpp
2. mkdir build
3. cd build
4. cmake -DLOADGEN_DIR=<path_to/>/loadgen -DLOADGEN_LIB_DIR=<path_to/>/loadgen/build  ..
5. make


# Running benchmark in Offline mode

First activate OpenVino environment:

```
source <OPENVINO_INSTALL_DIR/>bin/setupvars.sh
```

./cpp/bin/intel64/Debug/ov_mlperf -m <path_to_model/>/model.xml -data <path_to_mlperf_dir/>/build/preprocessed_data/preprocessed_files.pkl -mlperf_conf <path_to_mlperf_dir/>/build/mlperf.conf -user_conf <path_to_mlperf_dir/>/user.conf -scenario Offline -mode Accuracy -streams 20

For additional command line options use -h