"""
mlperf inference benchmarking tool
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import json
import logging
import os
import time
import sys

import tqdm
import numpy as np

import dataset
import imagenet
import coco

from more_itertools import chunked

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("main")

NANO_SEC = 1e9
MILLI_SEC = 1000

# pylint: disable=missing-docstring

# the datasets we support
SUPPORTED_DATASETS = {
    "imagenet": (
        imagenet.Imagenet,
        dataset.pre_process_vgg,
        dataset.PostProcessCommon(offset=0),
        {"image_size": [224, 224, 3]},
    ),
    "imagenet_mobilenet": (
        imagenet.Imagenet,
        dataset.pre_process_mobilenet,
        dataset.PostProcessArgMax(offset=-1),
        {"image_size": [224, 224, 3]},
    ),
    "imagenet_pytorch": (
        imagenet.Imagenet,
        dataset.pre_process_imagenet_pytorch,
        dataset.PostProcessArgMax(offset=0),
        {"image_size": [224, 224, 3]},
    ),
    "coco-300": (
        coco.Coco,
        dataset.pre_process_coco_mobilenet,
        coco.PostProcessCoco(),
        {"image_size": [300, 300, 3]},
    ),
    "coco-300-pt": (
        coco.Coco,
        dataset.pre_process_coco_pt_mobilenet,
        coco.PostProcessCocoPt(False, 0.3),
        {"image_size": [300, 300, 3]},
    ),
    "coco-1200": (
        coco.Coco,
        dataset.pre_process_coco_resnet34,
        coco.PostProcessCoco(),
        {"image_size": [1200, 1200, 3]},
    ),
    "coco-1200-onnx": (
        coco.Coco,
        dataset.pre_process_coco_resnet34,
        coco.PostProcessCocoPt(True, 0.05),
        {"image_size": [1200, 1200, 3], "use_label_map": True},
    ),
    "coco-1200-pt": (
        coco.Coco,
        dataset.pre_process_coco_resnet34,
        coco.PostProcessCocoPt(True, 0.05),
        {"image_size": [1200, 1200, 3], "use_label_map": True},
    ),
    "coco-1200-tf": (
        coco.Coco,
        dataset.pre_process_coco_resnet34,
        coco.PostProcessCocoTf(),
        {"image_size": [1200, 1200, 3], "use_label_map": False},
    ),
    #
    # furiosa golden pre/post-process
    #
    "imagenet-golden": (
        imagenet.Imagenet,
        dataset.pre_process_vgg,
        dataset.PostProcessArgMax(offset=0),
        {"image_size": [224, 224, 3]},
    ),
    "coco-300-golden": (
        coco.Coco,
        dataset.pre_process_coco_pt_mobilenet,
        coco.PostProcessCocoSSDMobileNetORT(False, 0.3),
        {"image_size": [300, 300, 3]},
    ),
    "coco-1200-golden": (
        coco.Coco,
        dataset.pre_process_coco_resnet34,
        coco.PostProcessCocoONNXNP(),
        {"image_size": [1200, 1200, 3], "use_label_map": False},
    ),
}
# pre-defined command line options so simplify things. They are used as defaults and can be
# overwritten from command line

SUPPORTED_PROFILES = {
    "defaults": {
        "dataset": "imagenet",
        "backend": "tensorflow",
        "cache": 0,
        "max-batchsize": 32,
    },
    # resnet
    "resnet50-tf": {
        "inputs": "input_tensor:0",
        "outputs": "ArgMax:0",
        "dataset": "imagenet",
        "backend": "tensorflow",
        "model-name": "resnet50",
    },
    "resnet50-onnxruntime": {
        "dataset": "imagenet",
        "outputs": "ArgMax:0",
        "backend": "onnxruntime",
        "model-name": "resnet50",
    },
    # mobilenet
    "mobilenet-tf": {
        "inputs": "input:0",
        "outputs": "MobilenetV1/Predictions/Reshape_1:0",
        "dataset": "imagenet_mobilenet",
        "backend": "tensorflow",
        "model-name": "mobilenet",
    },
    "mobilenet-onnxruntime": {
        "dataset": "imagenet_mobilenet",
        "outputs": "MobilenetV1/Predictions/Reshape_1:0",
        "backend": "onnxruntime",
        "model-name": "mobilenet",
    },
    # ssd-mobilenet
    "ssd-mobilenet-tf": {
        "inputs": "image_tensor:0",
        "outputs": "num_detections:0,detection_boxes:0,detection_scores:0,detection_classes:0",
        "dataset": "coco-300",
        "backend": "tensorflow",
        "model-name": "ssd-mobilenet",
    },
    "ssd-mobilenet-pytorch": {
        "inputs": "image",
        "outputs": "bboxes,labels,scores",
        "dataset": "coco-300-pt",
        "backend": "pytorch-native",
        "model-name": "ssd-mobilenet",
    },
    "ssd-mobilenet-onnxruntime": {
        "dataset": "coco-300",
        "outputs": "num_detections:0,detection_boxes:0,detection_scores:0,detection_classes:0",
        "backend": "onnxruntime",
        "data-format": "NHWC",
        "model-name": "ssd-mobilenet",
    },
    # ssd-resnet34
    "ssd-resnet34-tf": {
        "inputs": "image:0",
        "outputs": "detection_bboxes:0,detection_classes:0,detection_scores:0",
        "dataset": "coco-1200-tf",
        "backend": "tensorflow",
        "data-format": "NCHW",
        "model-name": "ssd-resnet34",
    },
    "ssd-resnet34-pytorch": {
        "inputs": "image",
        "outputs": "bboxes,labels,scores",
        "dataset": "coco-1200-pt",
        "backend": "pytorch-native",
        "model-name": "ssd-resnet34",
    },
    "ssd-resnet34-onnxruntime": {
        "dataset": "coco-1200-onnx",
        "inputs": "image",
        "outputs": "bboxes,labels,scores",
        "backend": "onnxruntime",
        "data-format": "NCHW",
        "max-batchsize": 1,
        "model-name": "ssd-resnet34",
    },
    "ssd-resnet34-onnxruntime-tf": {
        "dataset": "coco-1200-tf",
        "inputs": "image:0",
        "outputs": "detection_bboxes:0,detection_classes:0,detection_scores:0",
        "backend": "onnxruntime",
        "data-format": "NCHW",
        "model-name": "ssd-resnet34",
    },
    #
    # furiosa golden model setting
    #
    "ssd-resnet-golden": {
        "dataset": "imagenet-golden",
        "backend": "onnxruntime",
        "model-name": "resnet50",
    },
    "ssd-mobilenet-golden": {
        "dataset": "coco-300-golden",
        "backend": "onnxruntime",
        "data-format": "NCHW",
        "model-name": "ssd-mobilenet",
    },
    "ssd-resnet34-golden": {
        "dataset": "coco-1200-golden",
        "backend": "onnxruntime",
        "data-format": "NCHW",
        "model-name": "ssd-resnet34",
    },
    #
    # furiosa npu runtime backend setting
    #
    "resnet-golden-npu-legacy": {
        "inputs": "input_tensor:0",
        "outputs": "resnet_model/Squeeze:0_fused_dequantized",
        "dataset": "imagenet-golden",
        "backend": "npuruntime",
        "model-name": "resnet50",
    },
    "ssd-mobilenet-golden-npu-legacy": {
        "inputs": "image",
        "outputs": "class_logit_0_dequantized,class_logit_1_dequantized,class_logit_2_dequantized,"
        "class_logit_3_dequantized,class_logit_4_dequantized,class_logit_5_dequantized,"
        "box_regression_0_dequantized,box_regression_1_dequantized,box_regression_2_dequantized,"
        "box_regression_3_dequantized,box_regression_4_dequantized,box_regression_5_dequantized,",
        "dataset": "coco-300-golden",
        "backend": "npuruntime",
        "data-format": "NCHW",
        "model-name": "ssd-mobilenet",
    },
    "ssd-resnet34-golden-npu-legacy": {
        "inputs": "image:0",
        "outputs": "ssd1200/multibox_head/cls_0/BiasAdd:0_dequantized,"
        "ssd1200/multibox_head/cls_1/BiasAdd:0_dequantized,"
        "ssd1200/multibox_head/cls_2/BiasAdd:0_dequantized,"
        "ssd1200/multibox_head/cls_3/BiasAdd:0_dequantized,"
        "ssd1200/multibox_head/cls_4/BiasAdd:0_dequantized,"
        "ssd1200/multibox_head/cls_5/BiasAdd:0_dequantized,"
        "ssd1200/multibox_head/loc_0/BiasAdd:0_dequantized,"
        "ssd1200/multibox_head/loc_1/BiasAdd:0_dequantized,"
        "ssd1200/multibox_head/loc_2/BiasAdd:0_dequantized,"
        "ssd1200/multibox_head/loc_3/BiasAdd:0_dequantized,"
        "ssd1200/multibox_head/loc_4/BiasAdd:0_dequantized,"
        "ssd1200/multibox_head/loc_5/BiasAdd:0_dequantized",
        "dataset": "coco-1200-golden",
        "backend": "npuruntime",
        "data-format": "NCHW",
        "model-name": "ssd-resnet34",
    },
}

last_timeing = []


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS.keys(), help="dataset")
    parser.add_argument("--dataset-path", required=True, help="path to the dataset")
    parser.add_argument("--dataset-list", help="path to the dataset list")
    parser.add_argument("--data-format", choices=["NCHW", "NHWC"], help="data format")
    parser.add_argument("--profile", choices=SUPPORTED_PROFILES.keys(), help="standard profiles")
    parser.add_argument(
        "-b", "--max-batchsize", default=1, type=int, help="max batch size in a single inference"
    )
    parser.add_argument("--model", required=True, help="model file")
    parser.add_argument("--output", default="eval_result", help="test results")
    parser.add_argument("--inputs", help="model inputs")
    parser.add_argument("--outputs", help="model outputs")
    parser.add_argument("--backend", help="runtime to use")
    parser.add_argument("--model-name", help="name of the mlperf model, ie. resnet50")
    parser.add_argument("--threads", default=os.cpu_count(), type=int, help="threads")
    parser.add_argument("--qps", type=int, help="target qps")
    parser.add_argument("--cache", type=int, default=0, help="use cache")
    parser.add_argument(
        "--cache_dir", type=str, default=None, help="path to save preprocessed dataset"
    )
    parser.add_argument(
        "--accuracy", default=True, action="store_true", help="enable accuracy pass"
    )
    parser.add_argument(
        "--find-peak-performance", action="store_true", help="enable finding peak performance pass"
    )
    parser.add_argument("--debug", action="store_true", help="debug, turn traces on")

    # file to use mlperf rules compliant parameters
    parser.add_argument("--mlperf_conf", default="../../mlperf.conf", help="mlperf rules config")
    # file for user LoadGen settings such as target QPS
    parser.add_argument(
        "--user_conf",
        default="user.conf",
        help="user config for user LoadGen settings such as target QPS",
    )

    # below will override mlperf rules compliant settings - don't use for official submission
    parser.add_argument("--time", type=int, help="time to scan in seconds")
    parser.add_argument("-n", "--count", type=int, help="dataset items to use")
    parser.add_argument("--max-latency", type=float, help="mlperf max latency in pct tile")
    parser.add_argument(
        "--samples-per-query", type=int, help="mlperf multi-stream sample per query"
    )
    args = parser.parse_args()

    # don't use defaults in argparser. Instead we default to a dict, override that with a profile
    # and take this as default unless command line give
    defaults = SUPPORTED_PROFILES["defaults"]

    if args.profile:
        profile = SUPPORTED_PROFILES[args.profile]
        defaults.update(profile)
    for k, v in defaults.items():
        kc = k.replace("-", "_")
        if getattr(args, kc) is None:
            setattr(args, kc, v)
    if args.inputs:
        args.inputs = args.inputs.split(",")
    if args.outputs:
        args.outputs = args.outputs.split(",")

    return args


def get_backend(backend):
    if backend == "tensorflow":
        from backend_tf import BackendTensorflow

        backend = BackendTensorflow()
    elif backend == "onnxruntime":
        from backend_onnxruntime import BackendOnnxruntime

        backend = BackendOnnxruntime()
    elif backend == "null":
        from backend_null import BackendNull

        backend = BackendNull()
    elif backend == "pytorch":
        from backend_pytorch import BackendPytorch

        backend = BackendPytorch()
    elif backend == "pytorch-native":
        from backend_pytorch_native import BackendPytorchNative

        backend = BackendPytorchNative()
    elif backend == "tflite":
        from backend_tflite import BackendTflite

        backend = BackendTflite()
    elif backend == "npuruntime":
        from backend_npuruntime import BackendNPURuntime

        backend = BackendNPURuntime()
    else:
        raise ValueError("unknown backend: " + backend)
    return backend


class Item:
    """An item that we queue for processing by the thread pool."""

    def __init__(self, query_id, content_id, img, label=None):
        self.query_id = query_id
        self.content_id = content_id
        self.img = img
        self.label = label
        self.start = time.time()


class RunnerBase:
    def __init__(self, model, ds, threads, post_proc=None, max_batchsize=128):
        self.take_accuracy = False
        self.ds = ds
        self.model = model
        self.post_process = post_proc
        self.threads = threads
        self.take_accuracy = False
        self.max_batchsize = max_batchsize
        self.result_timing = []

    def handle_tasks(self, tasks_queue):
        pass

    def start_run(self, result_dict, take_accuracy):
        self.result_dict = result_dict
        self.result_timing = []
        self.take_accuracy = take_accuracy
        self.post_process.start()

    def run_one_item(self, qitem):
        # run the prediction
        try:
            results = self.model.predict({self.model.inputs[0]: qitem.img})
            processed_results = self.post_process(
                results, qitem.content_id, qitem.label, self.result_dict
            )
            if self.take_accuracy:
                self.post_process.add_results(processed_results)
                self.result_timing.append(time.time() - qitem.start)
        except Exception as ex:  # pylint: disable=broad-except
            src = [self.ds.get_item_loc(i) for i in qitem.content_id]
            log.error("thread: failed on contentid=%s, %s", src, ex)
            sys.exit(1)

    def enqueue(self, query_samples, pbar):
        query_id = idx = list(query_samples.keys())

        if len(query_samples) < self.max_batchsize:
            data, label = self.ds.get_samples(idx)
            self.run_one_item(Item(query_id, idx, data, label))
            pbar.update(len(query_samples))
        else:
            bs = self.max_batchsize
            for i in range(0, len(idx), bs):
                data, label = self.ds.get_samples(idx[i : i + bs])
                self.run_one_item(Item(query_id[i : i + bs], idx[i : i + bs], data, label))
                pbar.update(bs)

    def finish(self):
        pass


def add_results(final_results, name, count, result_dict, result_list, took, show_accuracy=False):
    percentiles = [50.0, 80.0, 90.0, 95.0, 99.0, 99.9]
    buckets = np.percentile(result_list, percentiles).tolist()
    buckets_str = ",".join(["{}:{:.4f}".format(p, b) for p, b in zip(percentiles, buckets)])

    if result_dict["total"] == 0:
        result_dict["total"] = len(result_list)

    # this is what we record for each run
    result = {
        "took": took,
        "mean": np.mean(result_list),
        "percentiles": {str(k): v for k, v in zip(percentiles, buckets)},
        "qps": len(result_list) / took,
        "count": count,
        "good_items": result_dict["good"],
        "total_items": result_dict["total"],
    }
    acc_str = ""
    if show_accuracy:
        result["accuracy"] = 100.0 * result_dict["good"] / result_dict["total"]
        acc_str = ", acc={:.3f}%".format(result["accuracy"])
        if "mAP" in result_dict:
            result["mAP"] = 100.0 * result_dict["mAP"]
            acc_str += ", mAP={:.3f}%".format(result["mAP"])

    # add the result to the result dict
    final_results[name] = result

    # to stdout
    print(
        "{} qps={:.2f}, mean={:.4f}, time={:.3f}{}, queries={}, tiles={}".format(
            name, result["qps"], result["mean"], took, acc_str, len(result_list), buckets_str
        )
    )


def main():
    global last_timeing
    args = get_args()

    log.info(args)

    # find backend
    backend = get_backend(args.backend)

    # override image format if given
    image_format = args.data_format if args.data_format else backend.image_format()

    # --count applies to accuracy mode only and can be used to limit the number of images
    # for testing. For perf model we always limit count to 200.
    count_override = False
    count = args.count

    # dataset to use
    wanted_dataset, pre_proc, post_proc, kwargs = SUPPORTED_DATASETS[args.dataset]
    ds = wanted_dataset(
        data_path=os.path.abspath(args.dataset_path),
        image_list=args.dataset_list,
        name=args.dataset,
        image_format=image_format,
        pre_process=pre_proc,
        use_cache=args.cache,
        cache_dir=args.cache_dir,
        count=count,
        **kwargs,
    )
    # load model to backend
    model = backend.load(args.model, inputs=args.inputs, outputs=args.outputs)
    final_results = {
        "runtime": model.name(),
        "version": model.version(),
        "time": int(time.time()),
        "cmdline": str(args),
    }

    if args.output:
        output_dir = os.path.abspath(args.output)
        os.makedirs(output_dir, exist_ok=True)
        os.chdir(output_dir)

    #
    # make one pass over the dataset to validate accuracy
    #
    count = ds.get_item_count()

    # warmup
    ds.load_query_samples([0])
    for _ in range(5):
        img, _ = ds.get_samples([0])
        _ = backend.predict({backend.inputs[0]: img})
    ds.unload_query_samples(None)

    scenario = "model evaluation"
    log.info("starting {}".format(scenario))
    runner = RunnerBase(
        model, ds, args.threads, post_proc=post_proc, max_batchsize=args.max_batchsize
    )
    result_dict = {"good": 0, "total": 0}
    runner.start_run(result_dict, args.accuracy)

    with tqdm.tqdm(total=count, unit="image") as pbar:
        for chunk in chunked(range(count), 1000):
            ds.load_query_samples(chunk)
            runner.enqueue(ds.image_list_inmemory, pbar)
            ds.unload_query_samples(None)

    last_timeing = runner.result_timing
    post_proc.finalize(result_dict, ds, output_dir=args.output)

    add_results(
        final_results,
        scenario,
        count,
        result_dict,
        last_timeing,
        time.time() - ds.last_loaded,
        args.accuracy,
    )

    runner.finish()

    #
    # write final results
    #
    file_name = os.path.basename(args.model).split(".onnx")[0]
    if args.output:
        with open(f"{file_name}_n={count}.json", "w") as f:
            json.dump(final_results, f, sort_keys=True, indent=4)


if __name__ == "__main__":
    main()
