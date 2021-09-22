# Resnet50 Int8 Calibration Steps

## Prepare Dataset for Calibration

The list of images used for calibration can be found at 
https://github.com/mlcommons/inference/blob/master/calibration/ImageNet/cal_image_list_option_2.txt

Download Imagenet ILSVRC2012 validation dataset from 
```bash
https://image-net.org/
```

Untar the file
```bash
$ mkdir imagenet2012
$ tar -xvf ILSVRC2012_img_val.tar -C imagenet2012
```

Run the script to arrange images into subfolders
```bash
$ bash calib_prep.sh
```

## Run int8 calibration

Set the path for calibration dataset
```bash
$ export DATA_PATH= path_to_calibration_dataset_folder
```

Set the path for MLPerf fp32 model for resnet50
```bash
$ export MODEL_PATH= path_to_mlperf_fp32_model
```

Perform int8 Calibration using IPEX
```bash
$ bash run_calibration.sh resnet50 $DATA_PATH $MODEL_PATH dnnl int8 jit resnet50_configure_jit_intel.json calibration
```

Run Accuracy check using the instructions for Accuracy of Resnet50 with newly generated JSON file

