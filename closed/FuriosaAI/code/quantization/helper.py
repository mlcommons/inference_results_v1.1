from typing import List, Dict, Tuple

import os

import json
import wget
import pathlib

import cv2
import onnx
import torch
import numpy as np
import tf2onnx

import furiosa_sdk_quantizer.frontend.onnx as furiosa_quantizer

from furiosa_sdk_quantizer.frontend.onnx.transformer.utils import rebuild_model
from mlperf_evaluation.tmp.ssd_mobilenet_v1 import MLCommons_SSDMobileNetV1


def edit_i8_model(model):
    new_nodes = []
    inputs = {model.graph.input[0].name: model.graph.input[0]}
    outputs = {out.name: out for out in model.graph.output}
    value_info = {vi.name: vi for vi in model.graph.value_info}
    for node in model.graph.node:
        if node.input[0] in inputs and node.op_type == "QuantizeLinear":
            continue
        if node.output[0] in outputs and node.op_type == "DequantizeLinear":
            continue
        new_nodes.append(node)

    for k, v in inputs.items():
        new_name = k + "_quantized"
        try:
            model.graph.input.remove(v)
            model.graph.value_info.remove(value_info[new_name])
            model.graph.input.insert(0, value_info[new_name])
        except KeyError:
            pass

    for k, v in outputs.items():
        new_name = k.split("_dequantized")[0] + "_quantized"
        try:
            model.graph.value_info.remove(value_info[new_name])
            model.graph.output.append(value_info[new_name])
            model.graph.output.remove(v)
        except KeyError:
            pass
    model = rebuild_model(model, new_nodes, eliminate=False)

    return model


def download_model(url):
    wget.download(url)

    if url.endswith(".zip"):
        filename = url.split("/")[-1]
        directory = filename.split(".zip")[0]

        from mlperf_evaluation.prepare import unzip

        unzip(filename, "./")

        for file in os.listdir(directory):
            if file.endswith(".pb"):
                os.rename(os.path.abspath(f"{directory}/{file}"), os.path.abspath(file))
                import shutil

                shutil.rmtree(os.path.abspath(directory))
                os.remove(os.path.abspath(filename))


def call_tf2onnx(
    protobuf_path: str, input_names: List[str], output_names: List[str], as_nchw=False
) -> onnx.ModelProto:
    graph_def, input_names, output_names = tf2onnx.tf_loader.from_graphdef(
        protobuf_path, input_names, output_names
    )
    if as_nchw:
        model_proto, _ = tf2onnx.convert.from_graph_def(
            graph_def,
            input_names=input_names,
            output_names=output_names,
            opset=12,
            inputs_as_nchw=input_names,
        )
    else:
        model_proto, _ = tf2onnx.convert.from_graph_def(
            graph_def, input_names=input_names, output_names=output_names, opset=12
        )
    return model_proto


def call_pth2onnx(
    pth_path: str,
    input_names: List[str],
    output_names: List[str],
    input_size: Tuple[int, int],
    save_path: str,
) -> None:
    model = MLCommons_SSDMobileNetV1()
    model.load_state_dict(torch.load(pth_path, map_location=torch.device("cpu")).state_dict())

    torch.onnx.export(
        model,
        torch.randn(1, 3, *input_size),
        save_path,
        opset_version=12,
        input_names=input_names,
        output_names=output_names,
    )


def optimize_model(path):
    model = onnx.load_model(path)
    opt_path = f'{path.split(".onnx")[0]}_optimized.onnx'
    opt_model = furiosa_quantizer.optimize_model(model)

    return opt_model


def attach_custom_ssd_detection_postprocess(
    model: onnx.ModelProto,
    ploc: str,
    plabel: str,
    anchors: np.array,
    feature_sizes: List[int],
    num_anchors: List[int],
    score_function: str,
    anchor_first: bool,
    num_classes: int,
    nms_score_threshold: float,
    nms_iou_threshold: float,
    max_detections: int,
    detections_per_class: int,
    scales: Tuple[float, float, float, float],
    dequantize: bool,
) -> onnx.ModelProto:
    """
    :param model: ONNX ModelProto
    :param ploc: predicted localization logits
    :param plabel: predicted label logits
    :param anchors: SSD model priors
    :param feature_sizes: Feature sizes of prediction heads
    :param num_anchors: Number of anchors per feature map
    :param score_function: sigmoid or softmax
    :param anchor_first: [1, len(anchors), 4] is expected for ploc as default. If anchor_first is False, it indicates
    ploc/plabel needs to be transposed.
    :param num_classes: Number of classes + background
    :param nms_score_threshold: float
    :param nms_iou_threshold: float
    :param max_detections: Maximum number of detections (boxes) to show.
    :param detections_per_class: Number of anchors used per class in Regular Non-Max-Suppression.
    :param scales: (y_scale, x_scale, h_scale, w_scale).
    :param dequantize: Whether to insert DequantizeLinear before Furiosa_Detection_PostProcess
    :return: ONNX ModelProto with ssd postprocessing
    """

    if score_function.lower() not in ["sigmoid", "softmax"]:
        raise Exception(f"Unknown score function: {score_function}")

    attrs = {
        "feature_sizes": feature_sizes,
        "num_anchors": num_anchors,
        "score_function": score_function.lower(),
        "anchor_first": bool(anchor_first),
        "num_classes": num_classes,
        "nms_score_threshold": nms_score_threshold,
        "nms_iou_threshold": nms_iou_threshold,
        "max_detections": max_detections,
        "detections_per_class": detections_per_class,
        "y_scale": scales[0],
        "x_scale": scales[1],
        "h_scale": scales[2],
        "w_scale": scales[3],
    }

    # make DequantizeLinear node/value_info for i8 model
    if dequantize:
        model.graph.node.extend(
            [make_dequantizelinear_node(ploc), make_dequantizelinear_node(plabel)]
        )
        output_shapes = {
            oup.name.split("_quantized")[0]: [
                dim.dim_value for dim in oup.type.tensor_type.shape.dim
            ]
            for oup in model.graph.output
        }
        model.graph.value_info.extend(
            [
                onnx.helper.make_tensor_value_info(
                    name=ploc + "_dequantized",
                    elem_type=onnx.TensorProto.FLOAT,
                    shape=output_shapes[ploc],
                ),
                onnx.helper.make_tensor_value_info(
                    name=plabel + "_dequantized",
                    elem_type=onnx.TensorProto.FLOAT,
                    shape=output_shapes[plabel],
                ),
            ]
        )
        ploc += "_dequantized"
        plabel += "_dequantized"

    ssd_detection_postprocess = onnx.helper.make_node(
        op_type="Furiosa_Detection_PostProcess",
        inputs=[ploc, plabel, "anchors"],
        outputs=["detection_boxes", "detection_classes", "detection_scores", "num_boxes"],
        **attrs,
    )

    model.graph.node.append(ssd_detection_postprocess)
    model.graph.initializer.append(
        onnx.numpy_helper.from_array(anchors.astype(np.float32), name="anchors")
    )
    model.graph.input.append(
        onnx.helper.make_tensor_value_info(
            name="anchors", elem_type=onnx.TensorProto.FLOAT, shape=anchors.shape
        )
    )
    model.graph.value_info.extend(model.graph.output)
    model.graph.ClearField("output")
    model.graph.output.extend(
        [
            onnx.helper.make_tensor_value_info(
                "detection_boxes", onnx.TensorProto.FLOAT, [1, max_detections, 4]
            ),
            onnx.helper.make_tensor_value_info(
                "detection_classes", onnx.TensorProto.FLOAT, [1, max_detections]
            ),
            onnx.helper.make_tensor_value_info(
                "detection_scores", onnx.TensorProto.FLOAT, [1, max_detections]
            ),
            onnx.helper.make_tensor_value_info(
                "num_boxes",
                onnx.TensorProto.FLOAT,
                [
                    1,
                ],
            ),
        ]
    )

    return model


def make_dequantizelinear_node(input_name):
    return onnx.helper.make_node(
        op_type="DequantizeLinear",
        inputs=[input_name + "_quantized", input_name + "_scale", input_name + "_zero_point"],
        outputs=[input_name + "_dequantized"],
    )


def calibrate_model(
    path: str, dataset: str, preproc: callable, save_path: str
) -> Dict[str, Tuple[float, float]]:
    # download calibration dataset if it does not exist
    if dataset == "imagenet":
        download_id = "1uZMo8xefLsB3Hl1A8GehZt_e9WbboxFg"
    elif dataset == "mscoco":
        download_id = "1VwWZcSaa5VoHHGg8qhgOnmOSO-0gidJH"
    else:
        raise Exception(f"Unknown dataset: {dataset}")

    calib_data_dir = f"{dataset}_calibration"
    if not os.path.exists(calib_data_dir):
        from mlperf_evaluation.prepare import process_dataset

        process_dataset(download_id, "tmp.tar.gz", calib_data_dir)

    # parse calibration files
    paths = []
    for ext in ["JPEG", "jpg"]:
        paths.extend([str(path) for path in pathlib.Path(calib_data_dir).glob(f"*.{ext}")])

    model = onnx.load_model(path)
    input_name = model.graph.input[0].name
    input_size = model.graph.input[0].type.tensor_type.shape.dim[-1].dim_value

    # preprocess calibration dataset
    dataset = []
    for path in paths:
        img = cv2.imread(path)
        preprocessed = preproc(img, [input_size, input_size, 3], need_transpose=True)
        dataset.append({input_name: np.expand_dims(preprocessed, axis=0)})
    assert len(dataset) == 500

    # calibrate
    dynamic_ranges = furiosa_quantizer.calibrate(model, dataset)

    # save dynamic ranges
    with open(save_path, "w") as f:
        json.dump(dynamic_ranges, f, indent=4)

    return dynamic_ranges


def quantize_model(path, dynamic_ranges):
    model = onnx.load_model(path)
    quant_model = furiosa_quantizer.quantize(model, True, True, 0, dynamic_ranges)
    quant_model = edit_i8_model(quant_model)

    return quant_model
