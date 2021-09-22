import logging
import os
import sys

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("BACKEND")

from baseBackend import baseBackend
"""
Resnet50 Pytorch SUT class
"""
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import intel_pytorch_extension as ipex
import _torch_ipex as core
import numpy as np

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
class Backend(baseBackend):
    def __init__(self, model_path = "", ipex=None, dnnl=True, jit= True,int8=True, configure_dir = "", **kwargs):
        if not os.path.isfile(model_path):
            log.error("Model not found: {}".format(model_path))
            sys.exit(1)
        if (int8 and not os.path.isfile(configure_dir)):
            log.error("Configure dir not found: {}".format(configure_dir))
            sys.exit(1)
        self.model_path = model_path
        self.ipex = ipex
        self.dnnl = dnnl
        self.configure_dir = configure_dir
        self.int8 = int8
        self.jit = jit
        self.model = models.__dict__['resnet50'](pretrained=True)
        print("Loaded pretrained model")
    def load_model(self):
        if self.ipex:
            import intel_pytorch_extension as ipex
        if self.dnnl:
            ipex.core.enable_auto_dnnl()
        else:
            self.core.disable_auto_dnnl()
        #self.model = models.__dict__['resnet50'](pretrained=True)
        print("model_path: " + self.model_path)
        self.model.load_state_dict(torch.load(self.model_path))
        log.info("Model loaded")
        #self.model = torch.nn.DataParallel(self.model)
        if self.ipex:
            self.model = self.model.to(device = ipex.DEVICE)
        if self.jit:
            self.model = torch.jit.script(self.model)
        if self.int8:
            self.conf = ipex.AmpConf(torch.int8, self.configure_dir)
        else:
            self.conf = ipex.AmpConf(None)
        self.model.eval()

    def predict(self, data):
        with torch.no_grad():
            with ipex.AutoMixPrecision(self.conf, running_mode="inference"):
                data = data.to(device = ipex.DEVICE)
                # compute output
                output = self.model(data)
        return output



