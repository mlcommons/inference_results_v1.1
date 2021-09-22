import argparse

import cv2
import numpy as np

from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType
from onnxruntime.quantization.calibrate import CalibrationMethod
from onnxruntime.quantization.quant_utils import QuantFormat

from dataset import pre_process_vgg


def parse_args():
    parser = argparse.ArgumentParser(description="ONNXRuntime quantization tool")
    parser.add_argument("--input", "-i", type=str)
    parser.add_argument("--output", "-o", type=str)
    parser.add_argument("--dataset", "-d", type=str)
    parser.add_argument("--entropy-calibration", default=False, action="store_true")
    return parser.parse_args()


# https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/tools/quantization/notebooks/imagenet_v2/mobilenet.ipynb
def preprocess_image(image_path, height, width, channels=3):
    image = cv2.imread(image_path)
    image_data = pre_process_vgg(image, dims=[height, width, channels], need_transpose=True)
    image_data = np.expand_dims(image_data, axis=0)
    return image_data


def preprocess_func(images_folder, height, width, size_limit=0):
    unconcatenated_batch_data = []
    import pathlib

    image_filepathes = [str(path) for path in pathlib.Path(images_folder).glob("*.JPEG")]
    for image_filepath in image_filepathes:
        # image_filepath = images_folder + '/' + image_name
        image_data = preprocess_image(image_filepath, height, width)
        unconcatenated_batch_data.append(image_data)
    batch_data = np.concatenate(np.expand_dims(unconcatenated_batch_data, axis=0), axis=0)
    return batch_data


image_height = 224
image_width = 224


class ResNetDataReader(CalibrationDataReader):
    def __init__(self, calibration_image_folder):
        self.image_folder = calibration_image_folder
        self.preprocess_flag = True
        self.enum_data_dicts = []
        self.datasize = 0

    def get_next(self):
        if self.preprocess_flag:
            self.preprocess_flag = False
            nhwc_data_list = preprocess_func(
                self.image_folder, image_height, image_width, size_limit=0
            )
            self.datasize = len(nhwc_data_list)
            self.enum_data_dicts = iter(
                [{"input_tensor:0": nhwc_data} for nhwc_data in nhwc_data_list]
            )
        return next(self.enum_data_dicts, None)


if __name__ == "__main__":
    args = parse_args()
    dr = ResNetDataReader(args.dataset)

    if args.entropy_calibration:
        method = CalibrationMethod.Entropy
    else:
        method = CalibrationMethod.MinMax

    quantize_static(
        args.input,
        args.output,
        dr,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        calibrate_method=method,
    )
