import os
import cv2
import logging
import re
import numpy as np
import sys
import torch.utils.data
import torch.utils.data.distributed

import torchvision.transforms as transforms

from OutputItem import OutputItem
from InputData import InputData
from PIL import Image

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("imagenet")

class Dataset():
    def __init__(self, data_path="", image_list=None, total_sample_count=1024, **kwargs):
        self.image_filenames = []
        self.label_list = []
        self.image_list_inmemory = {}
        self.label_list_inmemory = {}
        self.dims = (224, 224, 3)
        self.count = total_sample_count

        if image_list is None:
            image_list = os.path.join(data_path, "val_map.txt")

        if not os.path.isfile(image_list):
            log.error("image list not found: {}".format(image_list))
            sys.exit(1)

        not_found = 0
        with open(image_list, 'r') as f:
            for s in f:
                image_name, label = re.split(r"\s+", s.strip())
                src = os.path.join(data_path, image_name)
                if not os.path.exists(src):
                    # if the image does not exists ignore it
                    not_found += 1
                    continue
                self.image_filenames.append(src)
                self.label_list.append(int(label))

                # limit the dataset if requested
                if self.count and len(self.image_filenames) >= self.count:
                    break

        self.count = min(self.count, len(self.image_filenames))
        self.load_dataset_into_memory()

    def load_query_samples(self, sample_index_list):
        """
        Called by loadgen to load samples before sending queries to sut.
        Ideally complementary to load_dataset. If using this to load samples by loadgen, the samples are not necessarily available across processes - Needs to figure out if possible to work this out
        """
        pass

    def getCount(self):
        return self.counts

    def load_dataset_into_memory(self):
        """
        Responsible for loading all available dataset into memory.
        Ideally complementary to 'load_query_samples
        """
        log.info("Loading dataset into memory")
        for index in range(self.count):
            src = self.image_filenames[index]
            img_org = cv2.imread(src)
            processed = self.pre_process(img_org)
            self.image_list_inmemory[index] = processed
            self.label_list_inmemory[index] = self.label_list[index]

    def load_dataset(self):
        """
        Responsible for loading all available dataset into memory.
        Ideally complementary to 'load_query_samples
        """

    def unload_query_samples(self, sample_list):
        """
        Workload dependent. But typically not implemented if load_query_samples is not implemented
        """
        log.info("Called to unload data")
        pass

    def obj_unload_query_samples(self, sample_list):
        if sample_list:
            for sample in sample_list:
                if sample in self.image_list_inmemory :
                    del self.image_list_inmemory[sample]
                    del self.label_list_inmemory[sample]
        else:
            self.image_list_inmemory = {}
            self.label_list_inmemory = {}

    def get_samples(self, sample_index_list=[]):
        """
        Fetches and returns pre-processed data at requested 'sample_index_list'
        """
        outData = []
        for idx in sample_index_list:
            data = self.image_list_inmemory[idx]
            outData.append( data )

        #data = np.array(outData)
        data = torch.cat(outData, 0)
        return InputData(data=data, data_shape=data.shape)

    def get_warmup_samples(self):
        """
        Fetches and returns pre-processed data for warmup
        """
        import random
        num_samples = 10
        warmup_samples = []
        if len(self.image_list_inmemory) < num_samples:
            self.load_query_samples(list(range(num_samples)))

        sample_ids = random.choices(list(self.image_list_inmemory.keys()), k=num_samples)
        for idx in sample_ids:
            data_item = self.image_list_inmemory[idx]
            #data = np.array(data_item).reshape(1, 3, 224, 224)
            item = InputData(data=data_item, data_shape=data_item.shape)
            warmup_samples.append( item )

        return warmup_samples

    def pre_process(self, img):
        """
        Pre-processes a given input/image
        """
        from PIL import Image
        import torchvision.transforms.functional as F

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = F.resize(img, 256, Image.BILINEAR)
        img = F.center_crop(img, 224)
        img = F.to_tensor(img)
        img = F.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False)
        img = np.asarray(img, dtype='float32')
        img = torch.from_numpy(np.array([img], dtype='float32'))

        return img

    def post_process(self, query_ids, sample_index_list, results):
        """
        Post-processor that accepts loadgens query ids and corresponding inference output.
        post_process should return and OutputItem object which has two attributes:
        OutputItem.query_id_list
        OutputItem.results
        """
        processed_results = []
        results= results.to(torch.device('cpu'))
        results = results.numpy()
        results = np.argmax(results, axis=1)
        n = results.shape[0]
        for idx in range(n):
            result = results[idx]
            processed_results.append([result])
        return OutputItem(query_ids, processed_results, array_type_code='q')
