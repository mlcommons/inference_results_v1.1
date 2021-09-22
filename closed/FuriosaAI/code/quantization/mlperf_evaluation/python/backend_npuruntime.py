"""
tflite backend (https://github.com/tensorflow/tensorflow/lite)
"""

# pylint: disable=unused-argument,missing-docstring,useless-super-delegation

from threading import Lock

import furiosa
from furiosa.runtime import session

import backend


class BackendNPURuntime(backend.Backend):
    def __init__(self):
        super(BackendNPURuntime, self).__init__()
        self.sess = None
        self.lock = Lock()

    def version(self):
        return furiosa.__version__

    def name(self):
        return "furiosa-npu-runtime"

    def image_format(self):
        return "NCHW"

    def load(self, model_path, inputs=None, outputs=None):
        # there is no input/output meta data i the graph so it need to come from config.
        if not inputs:
            raise ValueError("BackendNPURuntime needs inputs")
        if not outputs:
            raise ValueError("BackendNPURuntime needs outputs")

        self.inputs = inputs
        self.outputs = outputs

        self.sess = session.create(str(model_path))
        return self

    def predict(self, feed):
        # assume 1 input
        key = [key for key in feed.keys()][0]
        outputs = self.sess.run(feed[key])
        return outputs.numpy()
