import sys
import os
import logging
import time

import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image

from utils import DefaultBoxes, Encoder, COCODetection, SSDTransformer
from ssd_r34 import SSD_R34

import intel_pytorch_extension as ipex
from baseBackend import baseBackend

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("SSD-Backend")

class CalibrateSSD(object):
    def __init__(self, calibration_list=None, calibration_data_path=None, dims=(1200, 1200)):
        if calibration_list is None or not os.path.exists(calibration_list):
            log.error("Calibration requires list of images")
            sys.exit(1)

        if  calibration_data_path is None or not os.path.isdir(calibration_data_path):
            log.error("Cannot find calibration_data_path {}".format(calibration_data_path))
            sys.exit(1)

        self.calibration_list = calibration_list
        self.data_path = calibration_data_path
        self.calibration_data = []
        self.size = dims
        
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])

        self.trans_val = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor(),
            #ToTensor(),
            self.normalize,])

        self.load_calibration_data()
        

    def load_calibration_data(self):
        log.info("Performing calibration")
        with open(self.calibration_list) as fid:
            names = fid.read().splitlines()
            for name in names:
                src = os.path.join(self.data_path, name)
                #if not os.path.exists(src):
                #    log.error("Could not find file {}".format(src))
                
                #img_org = cv2.imread(src)
                processed = self.pre_process(src)
                #data = np.array([ processed ])
                self.calibration_data.append( processed ) #torch.tensor(data, dtype=torch.float32) )
            
        self.cal_images_count = len(self.calibration_data)

    def resize(self, img):
        img = np.array(img, dtype=np.float32)
        if len(img.shape) < 3 or img.shape[2] != 3:
            # some images might be grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_height, im_width, _ = self.dims
        img = cv2.resize(img, (im_width, im_height), interpolation=cv2.INTER_LINEAR)

        return img
                

    def pre_process(self, img_path, need_transpose=True):
        img = Image.open(img_path).convert("RGB")
        img = self.trans_val(img).unsqueeze(0)
        return img


class Backend(baseBackend):
    def __init__(self, model_path=None, label_num=81, inputs=None, outputs=None, use_ipex=True, device="cpu", calibration=False, configuration_file="configure.json", calibration_list=None, calibration_data_path=None, use_bf16=False, num_calibration_iterations=128, use_jit=False, *kwargs):
        self.calibrate = calibration
        if model_path is None:
            log.error("Model path not provided")
            sys.exit(1)

        if not os.path.isfile(model_path):
            log.error("Cannot find model: {}".format(model_path))
            sys.exit(1)

        self.model_path = model_path
        self.inputs = inputs if inputs else []
        self.outputs = outputs if outputs else []
        self.ipex = use_ipex
        self.device = device
        self.bf16 = use_bf16
        self.cal_iters = num_calibration_iterations
        self.label_num = label_num
        self.jit = use_jit

        if self.calibrate:
            conf = ipex.AmpConf(torch.int8)
            self.load_model()
            calData = CalibrateSSD(calibration_list, calibration_data_path)
            log.info("Calibrating ssd-r34")
            step = 0
            while step < self.cal_iters:
                img = calData.calibration_data[ step % calData.cal_images_count]
                with torch.no_grad():
                    with ipex.AutoMixPrecision(conf, running_mode="calibration"):
                        inp = img.to(ipex.DEVICE)
                        start_time = time.time()
                        results = self.model(inp)
                        end_time = time.time()
                        if step % 10 == 0:
                            log.info("Calibration step {}".format(step))


                step += 1


            conf.save(configuration_file)
        
        if not os.path.isfile(configuration_file):
            log.error("Cannot find int8 configuration file {}".format(configuration_file))
            sys.exit(1)

        self.conf = ipex.AmpConf(torch.int8, configuration_file)

    def load_model(self):
        od = torch.load(self.model_path, map_location=lambda storage, loc: storage)
        self.model = SSD_R34(self.label_num)
        self.model.load_state_dict(od['model'])
        if self.ipex:
            import intel_pytorch_extension as ipex
            if self.bf16:
                ipex.enable_auto_mixed_precision(mixed_dtype=torch.bfloat16)
            #ipex.core.enable_auto_dnnl()
            self.model = self.model.to(ipex.DEVICE)

        if self.jit:
            self.model = torch.jit.script(self.model)

        self.model.eval()

        # find inputs from the model if not passed in by config
        if not self.inputs:
            self.inputs = []
            initializers = set()
            for i in self.model.graph.initializer:
                initializers.add(i.name)
            for i in self.model.graph.input:
                if i.name not in initializers:
                    self.inputs.append(i.name)

        # find outputs from the model if not passed in by config
        if not self.outputs:
            self.outputs = []
            for i in self.model.graph.output:
                self.outputs.append(i.name)

    def predict(self, input_data):
        
        with torch.no_grad():
            if self.ipex:
                #import intel_pytorch_extension as ipex
                
                #ipex.core.enable_auto_dnn()
                with ipex.AutoMixPrecision(self.conf, running_mode="inference"):
                    data = input_data.data.to(ipex.DEVICE)
                    output = self.model(data)
                #log.info("Output shape: {}".format(output.shape))
                
        return output
