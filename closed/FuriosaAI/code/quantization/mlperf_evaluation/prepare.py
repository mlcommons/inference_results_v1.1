import os

import gdown
import logging
import tarfile
import zipfile

from tools.upscale_coco import upscale_coco

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("prepare.py")


def untar(filename, extract_path):
    tar = tarfile.open(filename, "r:*")
    tar.extractall(path=extract_path)
    tar.close()


def unzip(filename, extract_path):
    with zipfile.ZipFile(filename, "r") as zip_ref:
        zip_ref.extractall(extract_path)


def gdownload(filename, id):
    url = URL.format(id)
    gdown.download(url, filename, quiet=False)


def process_dataset(id, filename, extract_path):
    if not os.path.exists(filename):
        print("download %s from google drive" % (filename))
        gdownload(filename=filename, id=id)
    else:
        print("%s exists. go to next step" % (filename))

    if filename.endswith("tar.gz"):
        print("untar %s..." % (filename))
        untar(filename, extract_path)
        print("move file(s) to %s" % (extract_path))
        print("done! remove tar file.")
        os.remove(filename)
    elif filename.endswith("zip"):
        print("unzip %s..." % (filename))
        unzip(filename, extract_path)
        print("move file(s) to %s" % (extract_path))
        print("done! remove tar file.")
        os.remove(filename)
    elif (
        filename.endswith("txt")
        or filename.endswith("pb")
        or filename.endswith("onnx")
        or filename.endswith("tflite")
    ):
        print("move %s to %s" % (filename, extract_path))
        if not os.path.exists(extract_path):
            import pathlib

            path = pathlib.Path(extract_path)
            path.mkdir(parents=True, exist_ok=True)
        os.rename(filename, os.path.join(extract_path, filename))
    else:
        raise Exception()


SUPPORTED_DATASETS = {
    "imagenet_dataset": {
        "id": "1Du3_-ZfbkUwCBY7hvdBgFjb14-kqlizo",
        "filename": "Data_val.tar.gz",
        "extract_path": "assets/ilsvrc",
        "final_path": "assets/ilsvrc/Data/CLS-LOC/val/ILSVRC2012_val_00000001.JPEG",
    },
    "imagenet_annotations": {
        "id": "1pIqrMCSsByWYll2_OsaVd8dSKUBaGChQ",
        "filename": "val_map.txt",
        "extract_path": "assets/ilsvrc/Data/CLS-LOC/val",
        "final_path": "assets/ilsvrc/Data/CLS-LOC/val/val_map.txt",
    },
    "mscoco_dataset": {
        "id": "1YN3u3ilMymGZrjrooleweR9-C7Vklkih",
        "filename": "mscoco_val2017.tar.gz",
        "extract_path": "assets/mscoco",
        "final_path": "assets/mscoco/val2017/000000000139.jpg",
    },
    "mscoco_annotations": {
        "id": "1XJRS0Ws6v14XsVcuRWg8rr6VW2SH_bPD",
        "filename": "mscoco_instances_val2017.tar.gz",
        "extract_path": "assets/mscoco",
        "final_path": "assets/mscoco/annotations/instances_val2017.json",
    },
    "imagenet_calibration_dataset": {
        "id": "1uZMo8xefLsB3Hl1A8GehZt_e9WbboxFg",
        "filename": "imagenet_calibration.tar.gz",
        "extract_path": "assets/calibration/imagenet_calibration",
        "final_path": "assets/calibration/imagenet_calibration/ILSVRC2012_val_00000033.JPEG",
    },
    "imagenet_calibration_dataset2": {
        "id": "1oCKZI8cFQkHD1meadXxCtknutMZXSSZ3",
        "filename": "imagenet_calibration2.tar.gz",
        "extract_path": "assets/calibration/imagenet_calibration2",
        "final_path": "assets/calibration/imagenet_calibration2/ILSVRC2012_val_00000066.JPEG",
    },
    "mscoco_calibration_dataset": {
        "id": "19Ce4M1JuygzihZk3lmFROXtABbgQ4GPb",
        "filename": "mscoco_calibration.tar.gz",
        "extract_path": "assets/calibration",
        "final_path": "assets/calibration/mscoco_calibration/000000000400.jpg",
    },
}

UPSCALE_COCO = [
    {
        "inputs": "assets/mscoco",
        "outputs": "assets/mscoco/upscale1200",
        "images": "val2017",
        "annotations": "annotations/instances_val2017.json",
        "size": (1200, 1200),
        "format": "png",
    },
    {
        "inputs": "assets/mscoco",
        "outputs": "assets/mscoco/upscale300",
        "images": "val2017",
        "annotations": "annotations/instances_val2017.json",
        "size": (300, 300),
        "format": "png",
    },
]

SUPPORTED_MODELS = {
    "mlcommons_resnet50": {
        "id": "1xIN1wjOnTVz3VR6Guk1Rr_yHRY7dTHQt",
        "filename": "mlcommons_resnet50_v1.5.onnx",
        "extract_path": "assets/models",
        "final_path": "assets/models/mlcommons_resnet50_v1.5.onnx",
    },
    "mlcommons_resnet50_fake_quant": {
        "id": "1WpsrXn4fifWtfwSKt2mHWsAt5ggMcUJf",
        "filename": "mlcommons_resnet50_v1.5_fake_quant.onnx",
        "extract_path": "assets/models",
        "final_path": "assets/models/mlcommons_resnet50_v1.5_fake_quant.onnx",
    },
    "mlcommons_resnet50_int8": {
        "id": "18vU6T1SKiAvMUmfekTBYPUQ9zxTi1sQX",
        "filename": "mlcommons_resnet50_v1.5_int8.onnx",
        "extract_path": "assets/models",
        "final_path": "assets/models/mlcommons_resnet50_v1.5_int8.onnx",
    },
    "ssd-mobilenet_v1": {
        "id": "1kOEpDj9_r-CqmyH0EOgA5-KIJmln7gmd",
        "filename": "mlcommons_ssd_mobilenet_v1.onnx",
        "extract_path": "assets/models/",
        "final_path": "assets/models/mlcommons_ssd_mobilenet_v1.onnx",
    },
    "ssd-mobilenet_v1_fake_quant": {
        "id": "1G68zFfBZt32puFb1h0JigMAYe7HqL0e5",
        "filename": "mlcommons_ssd_mobilenet_v1_fake_quant.onnx",
        "extract_path": "assets/models",
        "final_path": "assets/models/mlcommons_ssd_mobilenet_v1_fake_quant.onnx",
    },
    "ssd-mobilenet_v1_int8": {
        "id": "113Ppc6U1zdOQuyAlifLV7BKV1DzGwc61",
        "filename": "mlcommons_ssd_mobilenet_v1_int8.onnx",
        "extract_path": "assets/models",
        "final_path": "assets/models/mlcommons_ssd_mobilenet_v1_int8.onnx",
    },
    "ssd-resnet34": {
        "id": "1bHTuVFAsgr0EomNVdPhcOs6LoSTJTEyU",
        "filename": "mlcommons_ssd_resnet34.onnx",
        "extract_path": "assets/models/",
        "final_path": "assets/models/mlcommons_ssd_resnet34.onnx",
    },
    "ssd-resnet34_fake_quant": {
        "id": "1kJyX691jJawiYVNIYfXgk15eRHPEBChG",
        "filename": "mlcommons_ssd_resnet34_fake_quant.onnx",
        "extract_path": "assets/models",
        "final_path": "assets/models/mlcommons_ssd_resnet34_fake_quant.onnx",
    },
    "ssd-resnet34_int8": {
        "id": "113Ppc6U1zdOQuyAlifLV7BKV1DzGwc61",
        "filename": "mlcommons_ssd_resnet34_int8.onnx",
        "extract_path": "assets/models",
        "final_path": "assets/models/mlcommons_ssd_resnet34_int8.onnx",
    },
}

EXPERIMENT_MODELS = {
    "resnet": {
        "id": "1RDNIOY-o_e5m0CIgwYnm96Ci47hjhgPH",
        "filename": "resnet50_opset13.onnx",
        "extract_path": "assets/models/experiment",
        "final_path": "assets/models/experiment/resnet50_opset13.onnx",
    },
    "ssd-small": {
        "id": "1zKPSY07ZdQ9wBOUsPKWP9XI5u3UtwylT",
        "filename": "ssd_small_opset13.onnx",
        "extract_path": "assets/models/experiment",
        "final_path": "assets/models/experiment/ssd_small_opset13.onnx",
    },
    "ssd-large": {
        "id": "1hgqCElWqP0if3P4EqcECOIBfTsKhak7R",
        "filename": "ssd_large_opset13.onnx",
        "extract_path": "assets/models/experiment",
        "final_path": "assets/models/experiment/ssd_large_opset13.onnx",
    },
    "resnet-int8": {
        "id": "15Nxq5cDjNyuhA2W8OC8L5s-VQGDi10FW",
        "filename": "mlcommons_resnet50_v1.5_int8_legacy.onnx",
        "extract_path": "assets/models/legacy",
        "final_path": "assets/models/legacy/mlcommons_resnet50_v1.5_int8_legacy.onnx",
    },
    "ssd-small-int8": {
        "id": "1rdnogdlpCDeMIsuHJ4rjgLawrLWFkO7i",
        "filename": "mlcommons_ssd_mobilenet_v1_int8_legacy.onnx",
        "extract_path": "assets/models/legacy",
        "final_path": "assets/models/legacy/mlcommons_ssd_mobilenet_v1_int8_legacy.onnx",
    },
    "ssd-large-int8": {
        "id": "1dcuqi_H7vV5F6nJnu6Vs78eLqIu-4JFq",
        "filename": "mlcommons_ssd_resnet34_int8_legacy.onnx",
        "extract_path": "assets/models/legacy",
        "final_path": "assets/models/legacy/mlcommons_ssd_resnet34_int8_legacy.onnx",
    },
}
URL = "https://drive.google.com/uc?id={}"

log.info(SUPPORTED_DATASETS)


def main():
    print("\nprepare validataion datasets.. It will take a couple of minutes.")
    for key, value in SUPPORTED_DATASETS.items():
        print("download %s." % (key))
        id = value["id"]
        filename = value["filename"]
        extract_path = value["extract_path"]
        final_path = value["final_path"]

        if os.path.exists(final_path):
            print("%s already exists. skip processing dataset." % (final_path))
            continue

        process_dataset(id, filename, extract_path)
    print("done!")

    log.info(UPSCALE_COCO)

    for config in UPSCALE_COCO:
        inputs = config["inputs"]
        outputs = config["outputs"]
        images = config["images"]
        annotations = config["annotations"]
        size = config["size"]
        format = config["format"]

        print("\nprepare upscaled coco validataion dataset for evaluating SSD ResNet34.")
        if not os.path.exists(outputs):
            upscale_coco(inputs, outputs, images, annotations, size, format)
    print("done!")

    print("\nprepare protobuf models.. It will take a couple of minutes.")
    log.info(SUPPORTED_MODELS)

    for key, value in SUPPORTED_MODELS.items():
        id = value["id"]
        filename = value["filename"]
        extract_path = value["extract_path"]
        final_path = value["final_path"]

        if os.path.exists(final_path):
            print("%s already exists. skip processing models." % (final_path))
            continue

        process_dataset(id, filename, extract_path)
    print("done!")

    log.info(EXPERIMENT_MODELS)
    for key, value in EXPERIMENT_MODELS.items():
        id = value["id"]
        filename = value["filename"]
        extract_path = value["extract_path"]
        final_path = value["final_path"]

        if os.path.exists(final_path):
            print("%s already exists. skip processing models." % (final_path))
            continue

        process_dataset(id, filename, extract_path)
    print("done!")


if __name__ == "__main__":
    main()
