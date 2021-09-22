from typing import Optional, Text

import sys
import json
import argparse
import onnx

from torch.utils.data import DataLoader
from torchvision.datasets.folder import ImageFolder
from torchvision.datasets import ImageNet

import furiosa_sdk_quantizer.frontend.onnx
from furiosa_sdk_quantizer.evaluator.data_loader import random_subset
from furiosa_sdk_quantizer.evaluator.model_caller import ModelCaller
from furiosa_sdk_quantizer.frontend.onnx.quantizer import quantizer


def main():
    args = parse_args()
    if args.command == "export_spec":
        export_spec(args.input, args.output)
    elif args.command == "optimize":
        optimize(args.input, args.output)
    elif args.command == "build_calibration_model":
        build_calibration_model(args.input, args.output)
    elif args.command == "quantize":
        quantize(args.input, args.output, args.dynamic_ranges)
    elif args.command == "post_training_quantization_with_random_calibration":
        post_training_quantization_with_random_calibration(args.input, args.output, args.num_data)
    elif args.command == "calibrate_with_random":
        calibrate_with_random(args.input, args.output, args.num_data)
    elif args.command == "calibrate_with_data_loader":
        calibrate_with_data_loader(
            args.input,
            args.output,
            args.dataset_path,
            args.dataset_type,
            args.num_data,
            args.preprocess_from_onnx_model_exporter_registry,
        )
    elif args.command == "evaluate":
        evaluate(
            args.input,
            args.output,
            args.dataset_path,
            args.dataset_type,
            args.num_data,
            args.preprocess_from_onnx_model_exporter_registry,
            args.batch_size,
        )
    elif args.command == "evaluate_with_fake_quantization":
        evaluate_with_fake_quantization(
            args.input,
            args.output,
            args.dynamic_ranges,
            args.dataset_path,
            args.dataset_type,
            args.num_data,
            args.preprocess_from_onnx_model_exporter_registry,
            args.batch_size,
        )
    else:
        raise Exception(f"Unsupported command, {args.command}")


def parse_args():
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        "-i", "--input", type=str, help="Path to Model file (tflite, onnx are supported)"
    )
    common_parser.add_argument("-o", "--output", type=str, help="Path to Output file")

    dataset_parser = argparse.ArgumentParser(add_help=False)
    dataset_parser.add_argument("--dataset-path", type=str, help="Path to dataset")
    dataset_parser.add_argument("--dataset-type", type=str, help="Type of dataset")
    dataset_parser.add_argument("-n", "--num-data", type=int, help="The number of data")
    dataset_parser.add_argument("-p", "--preprocess-from-onnx-model-exporter-registry", type=str)

    parser = argparse.ArgumentParser(description="Furiosa AI quantizer")
    subparsers = parser.add_subparsers(dest="command")

    export_spec_cmd = subparsers.add_parser(
        "export_spec", help="export_spec help", parents=[common_parser]
    )

    build_calibration_model_cmd = subparsers.add_parser(
        "build_calibration_model", help="build calibrate model help", parents=[common_parser]
    )

    optimize_cmd = subparsers.add_parser("optimize", help="optimize help", parents=[common_parser])

    quantize_cmd = subparsers.add_parser("quantize", help="quantize help", parents=[common_parser])
    quantize_cmd.add_argument("-d", "--dynamic-ranges", type=str, help="Dynamic ranges")

    post_training_quantization_with_random_calibration = subparsers.add_parser(
        "post_training_quantization_with_random_calibration",
        help="calibrate help",
        parents=[common_parser],
    )
    post_training_quantization_with_random_calibration.add_argument(
        "-n", "--num-data", type=int, help="The number of random data"
    )

    calibrate_with_random_cmd = subparsers.add_parser(
        "calibrate_with_random", help="Output: dynamic ranges", parents=[common_parser]
    )
    calibrate_with_random_cmd.add_argument(
        "-n", "--num-data", type=int, help="The number of random data"
    )

    calibrate_cmd = subparsers.add_parser(
        "calibrate_with_data_loader", parents=[common_parser, dataset_parser]
    )

    evaluate_cmd = subparsers.add_parser("evaluate", parents=[common_parser, dataset_parser])
    evaluate_cmd.add_argument("-b", "--batch-size", type=int)

    evaluate_with_fake_quantization_cmd = subparsers.add_parser(
        "evaluate_with_fake_quantization", parents=[common_parser, dataset_parser]
    )
    evaluate_with_fake_quantization_cmd.add_argument("-b", "--batch-size", type=int)
    evaluate_with_fake_quantization_cmd.add_argument(
        "-d", "--dynamic-ranges", type=str, help="Dynamic ranges)"
    )

    return parser.parse_args()


def export_spec(input: Optional[Text] = None, output: Optional[Text] = None):
    model = _read_model(input)
    if output is not None:
        with open(output, "w") as writable:
            furiosa_sdk_quantizer.frontend.onnx.export_spec(model, writable)
    else:
        furiosa_sdk_quantizer.frontend.onnx.export_spec(model, sys.stdout)


def optimize(input: Optional[Text] = None, output: Optional[Text] = None):
    model = _read_model(input)
    model = furiosa_sdk_quantizer.frontend.onnx.optimize_model(model)
    if output is not None:
        onnx.save_model(model, output)
    else:
        onnx.save_model(model, sys.stdout)


def build_calibration_model(input: Optional[Text] = None, output: Optional[Text] = None):
    model = _read_model(input)
    model = furiosa_sdk_quantizer.frontend.onnx.build_calibration_model(model)
    if output is not None:
        onnx.save_model(model, output)
    else:
        onnx.save_model(model, sys.stdout)


def quantize(
    input: Optional[Text] = None, output: Optional[Text] = None, dynamic_ranges: str = None
):
    model = _read_model(input)
    model = furiosa_sdk_quantizer.frontend.onnx.optimize_model(model)
    with open(dynamic_ranges, "r") as readable:
        dynamic_ranges = json.load(readable)
    model = furiosa_sdk_quantizer.frontend.onnx.quantize(
        model,
        per_channel=True,
        static=True,
        mode=quantizer.QuantizationMode.dfg,
        dynamic_ranges=dynamic_ranges,
    )
    if output is not None:
        onnx.save_model(model, output)
    else:
        onnx.save_model(model, sys.stdout)


def _read_model(input: Optional[Text] = None) -> onnx.ModelProto:
    if input is not None:
        with open(input, "rb") as readable:
            model = onnx.load_model(readable, onnx.helper.ModelProto)
    else:
        model = onnx.load_model(sys.stdin, onnx.helper.ModelProto)

    return model


def _load_dataset(
    dataset_path: str,
    dataset_type: str,
    num_data: int,
    preprocess_from_registry: str,
    batch_size: int,
) -> DataLoader:
    # FIXME: onnx-model-exporter takes too long to load
    from onnx_model_exporter.models import registry

    model_cls = registry.model_entrypoint(preprocess_from_registry)
    _, transform = ModelCaller(model_cls, "onnx").call()

    # TODO: support various type of dataset
    if dataset_type == "ImageFolder":
        dataset = ImageFolder(dataset_path, transform)
    elif dataset_type == "ImageNetValDataset":
        dataset = ImageNet(dataset_path, split="val", transform=transform)
    else:
        raise ValueError(f"Unexpected dataset type: {dataset_type}.")

    if num_data == 0:
        num_data = len(dataset)

    # `seed` is fixed to 1 because we need to get the same subset of
    # `dataset` across multiple executions.
    dataset = random_subset(dataset, num_data, seed=1)
    # The `shuffle` argument of DataLoader.__init__ does not have to be
    # set to True because we are calibrating or evaluating, not
    # training, the model.
    loader = DataLoader(dataset, batch_size)

    return loader


def post_training_quantization_with_random_calibration(
    input: Optional[Text] = None, output: Optional[Text] = None, num_data: Optional[int] = None
):
    model = _read_model(input)
    model = furiosa_sdk_quantizer.frontend.onnx.post_training_quantization_with_random_calibration(
        model,
        static=True,
        per_channel=True,
        mode=quantizer.QuantizationMode.dfg,
        num_data=num_data,
    )
    if output is not None:
        onnx.save_model(model, output)
    else:
        onnx.save_model(model, sys.stdout)


def calibrate_with_random(
    input: Optional[Text] = None, output: Optional[Text] = None, num_data: Optional[int] = None
):
    model = _read_model(input)
    dynamic_ranges = furiosa_sdk_quantizer.frontend.onnx.calibrate_with_random(model, num_data)
    if output is not None:
        with open(output, "w") as f:
            json.dump(dynamic_ranges, f, ensure_ascii=True, indent=2)
    else:
        json.dump(dynamic_ranges, sys.stdout, ensure_ascii=True, indent=2)


def calibrate_with_data_loader(
    input: Optional[Text],
    output: Optional[Text],
    dataset_path: Optional[Text],
    dataset_type: Text,
    num_data: Optional[int],
    preprocess: Optional[Text],
) -> None:
    model = _read_model(input)
    loader = _load_dataset(dataset_path, dataset_type, num_data, preprocess, 1)
    dynamic_ranges = furiosa_sdk_quantizer.frontend.onnx.calibrate_with_data_loader(model, loader)
    if output is not None:
        with open(output, "w") as f:
            json.dump(dynamic_ranges, f, ensure_ascii=True, indent=2)
    else:
        json.dump(dynamic_ranges, sys.stdout, ensure_ascii=True, indent=2)


def evaluate(
    input: Optional[Text],
    output: Optional[Text],
    dataset_path: Optional[Text],
    dataset_type: Text,
    num_data: Optional[int],
    preprocess: Optional[Text],
    batch_size: Optional[int] = 1,
) -> None:
    import furiosa_sdk_quantizer.evaluator.eval_imagenet

    model = _read_model(input)
    loader = _load_dataset(dataset_path, dataset_type, num_data, preprocess, batch_size)
    accuracy = furiosa_sdk_quantizer.evaluator.eval_imagenet.evaluate(model, loader)
    if output is not None:
        with open(output, "w") as f:
            json.dump(accuracy, f, ensure_ascii=True, indent=2)
    else:
        json.dump(accuracy, sys.stdout, ensure_ascii=True, indent=2)


def evaluate_with_fake_quantization(
    input: Optional[Text],
    output: Optional[Text],
    dynamic_ranges: str,
    dataset_path: Optional[Text],
    dataset_type: Text,
    num_data: Optional[int],
    preprocess: Optional[int],
    batch_size: Optional[int] = 1,
) -> None:
    import furiosa_sdk_quantizer.evaluator.eval_imagenet

    with open(dynamic_ranges) as readable:
        dynamic_ranges = json.load(readable)

    model = _read_model(input)
    model = furiosa_sdk_quantizer.frontend.onnx.optimize_model(model)
    model = furiosa_sdk_quantizer.frontend.onnx.quantize(
        model,
        per_channel=True,
        static=True,
        mode=quantizer.QuantizationMode.fake,
        dynamic_ranges=dynamic_ranges,
    )

    loader = _load_dataset(dataset_path, dataset_type, num_data, preprocess, batch_size)
    accuracy = furiosa_sdk_quantizer.evaluator.eval_imagenet.evaluate(model, loader)
    if output is not None:
        with open(output, "w") as f:
            json.dump(accuracy, f, ensure_ascii=True, indent=2)
    else:
        json.dump(accuracy, sys.stdout, ensure_ascii=True, indent=2)
