import json
import logging
import os
import time


import cv2
from PIL import Image
import numpy as np
from pycocotools.cocoeval import COCOeval
#import pycoco
import torch
import torchvision.transforms as transforms

from utils import DefaultBoxes, Encoder, COCODetection, SSDTransformer

from InputData import InputData
from OutputItem import OutputItem

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("SSDR34-Coco")

def dboxes_R34_coco(figsize,strides):
    feat_size = [[50, 50], [25, 25], [13, 13], [7, 7], [3, 3], [3, 3]]
    steps=[(int(figsize[0]/fs[0]),int(figsize[1]/fs[1])) for fs in feat_size]
    scales = [(int(s*figsize[0]/300),int(s*figsize[1]/300)) for s in [21, 45, 99, 153, 207, 261, 315]]
    aspect_ratios =  [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    dboxes = DefaultBoxes(figsize, feat_size, steps, scales, aspect_ratios)
    return dboxes


class Dataset(object):
    def __init__(self, data_path=None, annotations_file=None, name=None, use_cache=0, image_size=None, dims=(1200, 1200, 3),
                 image_format="NHWC", total_sample_count=256, use_label_map=False, score_threshold=0.01, device=0, **kwargs):
        super().__init__()
        self.image_size = image_size
        self.image_list = []
        self.label_list = []
        self.image_ids = []
        self.image_sizes = []
        self.count = total_sample_count
        self.use_cache = use_cache
        self.data_path = data_path
        self.use_label_map=use_label_map
        self.score_threshold = score_threshold
        self.dims = dims
        self.size = dims[:2]
        self.images_in_memory = {}

        # input images are in HWC
        self.need_transpose = True if image_format == "NCHW" else False
        not_found = 0 
        empty_80catageories = 0
        if annotations_file is None:
            # by default look for val_map.txt
            annotations_file = os.path.join(data_path, "annotations/instances_val2017.json")
        self.annotations_file = annotations_file

        if self.use_label_map:
            # for pytorch
            label_map = {}
            with open(self.annotation_file) as fin:
                annotations = json.load(fin)
            for cnt, cat in enumerate(annotations["categories"]):
                label_map[cat["id"]] = cnt + 1

        #os.makedirs(self.cache_dir, exist_ok=True)
        start = time.time()
        images = {}
        with open(self.annotations_file, "r") as f:
            coco = json.load(f)

        for i in coco["images"]:
            images[i["id"]] = {"file_name": i["file_name"],
                               "height": i["height"],
                               "width": i["width"],
                               "bbox": [],
                               "category": []}

        for a in coco["annotations"]:
            i = images.get(a["image_id"])
            if i is None:
                continue
            catagory_ids = label_map[a.get("category_id")] if self.use_label_map else a.get("category_id")
            i["category"].append(catagory_ids)
            i["bbox"].append(a.get("bbox"))

        for image_id, img in images.items():
            image_name = os.path.join("val2017", img["file_name"])
            src = os.path.join(data_path, image_name)
            if not os.path.exists(src):
                # if the image does not exists ignore it
                #log.info("Image source not exist: {}".format(src))
                not_found += 1
                continue
            if len(img["category"])==0 and self.use_label_map: 
                #if an image doesn't have any of the 81 categories in it    
                empty_80catageories += 1 #should be 48 images - thus the validation sert has 4952 images
                continue 

            self.image_ids.append(image_id)
            self.image_list.append(src) #(image_name)
            self.image_sizes.append((img["height"], img["width"]))
            self.label_list.append((img["category"], img["bbox"]))

            # limit the dataset if requested
            if self.count and len(self.image_list) >= self.count:
                break

        self.count = min(self.count, len(self.image_list))

        time_taken = time.time() - start
        if not self.image_list:
            log.error("no images in image list found")
            raise ValueError("no images in image list found")
        if not_found > 0:
            log.info("reduced image list, %d images not found", not_found)
        if empty_80catageories > 0:
            log.info("reduced image list, %d images without any of the 80 categories", empty_80catageories)

        log.info("loaded {} images, cache={}, took={:.1f}sec".format(
            len(self.image_list), use_cache, time_taken))

        self.label_list = np.array(self.label_list)
        
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])
        self.trans_val = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor(),
            self.normalize,])

        dboxes = dboxes_R34_coco((1200,1200),[3,3,2,2,2,2])
        self.encoder = Encoder(dboxes)
        self.device = device

    def load_dataset_into_memory(self):
        j = 0
        while j < self.count:
            src = self.image[j]
            img_org = cv2.imread(src)
            processed = self.pre_process(img_org)
            data = torch.tensor(processed, dtype=torch.float32)
            self.images_in_memory[j] = data
            j += 1


    def load_dataset(self):
        j = 0
        
        while j < self.count:
            src = self.image_list[j]
            processed = self.pre_process(src)
            self.images_in_memory[j] = processed
            j += 1


    def load_query_samples(self, sample_list):
        pass

    def unload_query_samples(self, sample_list):
        if sample_list:
            for sample in sample_list:
                if sample in self.images_in_memory :
                    del self.images_in_memory[sample]
        else:
            self.images_in_memory = {}

    def get_warmup_samples(self):
        import random
        num_samples = 10
        samples = []
        sample_ids = random.choices(list(self.images_in_memory.keys()), k=num_samples)
        for j in sample_ids:
            data = self.images_in_memory[j]
            item = InputData(data=data)
            samples.append(item)

        return samples

    def get_samples(self, sample_index_list):
        data = torch.cat([self.images_in_memory[id] for id in sample_index_list], 0)
        return InputData(data=data)


    def get_item(self, nr):
        """Get image by number in the list."""
        dst = os.path.join(self.cache_dir, self.image_list[nr])
        img = np.load(dst + ".npy")
        return img, self.label_list[nr]

    def get_item_loc(self, nr):
        src = os.path.join(self.data_path, self.image_list[nr])
        return src

    
    def pre_process(self, img_path, need_transpose=True):
        img = Image.open(img_path).convert("RGB")
        img = self.trans_val(img).unsqueeze(0)
        return img

    def post_process(self, query_ids, sample_index_list, results):
        results = self.encoder.decode_batch(results[0], results[1], 0.50, 200,device=self.device)

        processed_results = []
        for idx, result in enumerate(results):
            detection_boxes, detection_classes, scores = [r.cpu().numpy() for r in result]
            
            
            detected_objects = []
            for box, detection_class, score in zip(detection_boxes, detection_classes, scores):
                # comes from model as:  0=xmax 1=ymax 2=xmin 3=ymin
                detected_objects.append( [float(sample_index_list[idx]),
                                              box[1], box[0], box[3], box[2],
                                              score,
                                              float(detection_class)])
            processed_results.append( np.array(detected_objects, np.float32).tobytes() )
        return OutputItem(query_ids, processed_results, array_type_code='B')
