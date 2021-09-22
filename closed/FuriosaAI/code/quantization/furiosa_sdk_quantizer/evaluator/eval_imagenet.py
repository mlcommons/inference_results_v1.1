from typing import Dict, Union
import os
import torch
import tqdm
import onnx

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder, ImageNet

from furiosa_sdk_quantizer.evaluator.model_caller import ModelCaller
from furiosa_sdk_quantizer.evaluator.data_loader import random_subset
from furiosa_sdk_quantizer.evaluator.model_executor import ModelExecutor
from furiosa_sdk_quantizer.evaluator.evaluation_metric import ClassificationAccuracy

os.environ["KMP_DUPLICATE_LIB_OK"] = "Tru"


def run_eval(
    model_class,
    model_type,
    val_dir,
    cal_dir=None,
    num_eval=100,
    batch_size=1,
    quantize=False,
    is_print=False,
) -> None:
    caller = ModelCaller(model_class, model_type)
    model, transform = caller.call()

    if model_type == "onnx":
        from furiosa_sdk_quantizer.frontend.onnx import optimize_model

        model = optimize_model(model)

    if model_type == "onnx" and quantize:
        from furiosa_sdk_quantizer.frontend.onnx import (
            build_calibration_model,
            ONNXCalibrator,
            quantize,
        )

        calibration_model = build_calibration_model(model)
        calibration_dataset = ImageFolder(cal_dir, transform=model_class.model_config["transform"])
        dynamic_ranges = ONNXCalibrator(
            calibration_model,
        ).calibrate_with_data_loader(DataLoader(calibration_dataset))
        # set mode=1 <==> simulated quantization
        model = quantize(
            model, per_channel=True, static=True, mode=1, dynamic_ranges=dynamic_ranges
        )

    num_params = caller.param_count
    num_macs = caller.mac_count
    if is_print:
        print(
            f"model {caller.model_name} called.\n"
            f"\tparam count: {num_params:,}\n"
            f"\tmac_count: {num_macs:,}\n"
        )

    dataset = ImageNet(val_dir, split="val", transform=transform)
    # `seed` is fixed to 1 because we need to get the same subset of
    # `dataset` across multiple executions.
    dataset = random_subset(dataset, num_eval, seed=1)
    # The `shuffle` argument of DataLoader.__init__ does not have to be
    # set to True because we are evaluating, not training, the model.
    loader = DataLoader(dataset, batch_size)

    if is_print:
        print(f"backend: {model_type}")
        print(f"feed {num_eval} samples with batch_size {batch_size}.\n")

    accuracy = evaluate(model, loader, is_print)

    return {"n_params": num_params, "n_macs": num_macs, **accuracy}


def evaluate(
    model: Union[onnx.ModelProto, torch.nn.Module], loader: DataLoader, is_print: bool = False
) -> Dict[str, float]:
    executor = ModelExecutor(model)
    metric = ClassificationAccuracy()

    for input, target in tqdm.tqdm(loader):
        pred = executor.feed(input)
        metric.measure(pred, target)

    if is_print:
        print("eval results:")
        metric.print_result()
        print("end of eval.")

    accuracy = metric.announce()

    return {
        "top1_acc": accuracy["top1"],
        "top5_acc": accuracy["top5"],
    }
