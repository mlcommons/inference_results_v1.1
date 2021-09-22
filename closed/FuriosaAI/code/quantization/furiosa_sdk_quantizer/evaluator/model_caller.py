# FIXME, Uncomment after resolving dependency issue
# from onnx_model_exporter.models.arch_model import ModelBase


class ModelCaller:
    # FIXME, Uncomment after resolving dependency issue
    # def __init__(self, model_class: ModelBase, model_type: str):
    def __init__(self, model_class, model_type: str):
        self.model_class = model_class
        self.model_type = model_type
        self.transform = self.model_class.model_config["transform"]
        self.model = None
        self.macs = None
        self.params = None

    def call(self):
        if not self.model_class.has_pretrained:
            return None
        self.model = self.model_class(pretrained=True)

        from ptflops import get_model_complexity_info

        self.macs, self.params = get_model_complexity_info(
            self.model.pytorch_model,
            self.model_class.input_shape,
            verbose=False,
            print_per_layer_stat=False,
            as_strings=False,
        )

        if self.model_type == "pytorch":
            model = self.model.pytorch_model
            model.eval()
        elif self.model_type == "onnx":
            model = self.model.onnx_model
        else:
            raise Exception("Unknown model_type: %s" % self.model_type)

        return model, self.transform

    @property
    def param_count(self):
        return self.params

    @property
    def mac_count(self):
        return int(self.macs)

    @property
    def model_name(self):
        return self.model_class.model_name
