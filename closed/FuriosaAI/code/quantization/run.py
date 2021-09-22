import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))


def set_module_path(project_root):
    for module_path in ["furiosa_quantizer", "mlperf_evaluation"]:
        path = os.path.join(project_root, module_path)
        if path not in sys.path:
            sys.path.append(path)


set_module_path(PROJECT_ROOT)

from helper import *
from mlperf_evaluation.python.dataset import (
    pre_process_vgg,
    pre_process_coco_pt_mobilenet,
    pre_process_coco_resnet34,
)
from mlperf_evaluation.tmp.ssd_resnet34_postprocess import dboxes_R34_coco
from mlperf_evaluation.tmp.ssd_mobilenet_v1 import MLCommons_SSDMobileNetV1

PROFILES = {
    "resnet": {
        "url": "https://zenodo.org/record/2535873/files/resnet50_v1.pb",
        "download_path": "resnet50_v1.pb",
        "model_name": "mlcommons_resnet50_v1.5",
        "dataset": "imagenet",
        "preproc": pre_process_vgg,
        "input_names": ["input_tensor:0"],
        "output_names": ["ArgMax:0"],
    },
    "ssd-small": {
        "url": "https://zenodo.org/record/3239977/files/ssd_mobilenet_v1.pytorch",
        "download_path": "ssd_mobilenet_v1.pytorch",
        "model_name": "mlcommons_ssd_mobilenet_v1",
        "dataset": "mscoco",
        "preproc": pre_process_coco_pt_mobilenet,
        "input_names": ["image"],
        "output_names": ["ssd300/concat", "ssd300/concat_1"],
        "input_size": (300, 300),
        "postprocessing": {
            "ploc": "ssd300/concat_1",
            "plabel": "ssd300/concat",
            "anchors": MLCommons_SSDMobileNetV1().priors.numpy(),
            "score_function": "sigmoid",
            "anchor_first": True,
            "num_classes": 91,
            "nms_score_threshold": 0.3,
            "nms_iou_threshold": 0.6,
            "max_detections": 100,
            "detections_per_class": 100,
            "feature_sizes": [19, 19, 10, 10, 5, 5, 3, 3, 2, 2, 1, 1],
            "num_anchors": [3, 6, 6, 6, 6, 6],
            "scales": [10.0, 10.0, 5.0, 5.0],
        },
    },
    "ssd-large": {
        "url": "https://zenodo.org/record/3345892/files/tf_ssd_resnet34_22.1.zip",
        "download_path": "resnet34_tf.22.1.pb",
        "model_name": "mlcommons_ssd_resnet34",
        "dataset": "mscoco",
        "preproc": pre_process_coco_resnet34,
        "input_names": ["image:0"],
        "output_names": ["ssd1200/concat:0", "ssd1200/concat_1:0"],
        "postprocessing": {
            "ploc": "ssd1200/concat_1:0",
            "plabel": "ssd1200/concat:0",
            "anchors": dboxes_R34_coco()(order="ltrb"),
            "score_function": "softmax",
            "anchor_first": False,
            "num_classes": 81,
            "nms_score_threshold": 0.05,
            "nms_iou_threshold": 0.45,
            "max_detections": 200,
            "detections_per_class": 200,
            "feature_sizes": [50, 50, 25, 25, 13, 13, 7, 7, 3, 3, 3, 3],
            "num_anchors": [4, 6, 6, 6, 4, 4],
            "scales": [1.0, 1.0, 2.0, 2.0],
        },
    },
}

if __name__ == "__main__":
    try:
        key = sys.argv[1]
    except:
        print("python run.py <key>")
        print("<key> must be one of [resnet, ssd-small, ssd-large]")

    profile = PROFILES[key]
    model_name = profile["model_name"]
    model_path = f"{model_name}.onnx"
    dynamic_range_path = f"{model_name}_dynamic_ranges.json"

    # download original model listed on
    # https://github.com/mlcommons/inference/tree/master/vision/classification_and_detection
    if not os.path.exists(profile["download_path"]):
        download_model(profile["url"])

    # convert original model to onnx
    orig_model_path = profile["download_path"]
    input_names = profile["input_names"]
    output_names = profile["output_names"]
    if not os.path.exists(model_path):
        if key in ["resnet", "ssd-large"]:
            as_nchw = False
            if key == "resnet":
                as_nchw = True
            onnx_model = call_tf2onnx(orig_model_path, input_names, output_names, as_nchw)
            onnx.save_model(onnx_model, model_path)
        else:
            call_pth2onnx(
                orig_model_path, input_names, output_names, tuple(profile["input_size"]), model_path
            )

    # optimize onnx model
    opt_path = f"{model_name}_optimized.onnx"
    if not os.path.exists(opt_path):
        opt_model = optimize_model(model_path)
        onnx.save_model(opt_model, opt_path)
    else:
        opt_model = onnx.load_model(opt_path)

    # calibrate
    if not os.path.exists(dynamic_range_path):
        dynamic_ranges = calibrate_model(
            opt_path, profile["dataset"], profile["preproc"], dynamic_range_path
        )
    else:
        with open(dynamic_range_path, "r") as f:
            dynamic_ranges = json.load(f)

    # quantize
    quant_path = f"{model_name}_int8.onnx"
    if not os.path.exists(quant_path):
        quant_model = quantize_model(opt_path, dynamic_ranges)
        # attach Furiosa_Detection_PostProcess for SSD-Small/Large
        try:
            quant_model = attach_custom_ssd_detection_postprocess(
                quant_model, **profile["postprocessing"], dequantize=True
            )
        except KeyError:
            pass
        onnx.save_model(quant_model, quant_path)
    else:
        quant_model = onnx.load_model(quant_path)
