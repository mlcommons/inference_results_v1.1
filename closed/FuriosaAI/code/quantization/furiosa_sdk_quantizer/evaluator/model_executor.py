from typing import Union

import onnx
import torch
import onnxruntime as ort


class ModelExecutor:
    def __init__(self, model: Union[onnx.ModelProto, torch.nn.Module]):
        self.model = model
        if isinstance(model, torch.nn.Module):
            self.feed = self.pytorch_backend
        elif isinstance(model, onnx.ModelProto):
            self.feed = self.onnxruntime_backend
        else:
            raise Exception(f"Unknown model type: {model}")

    def pytorch_backend(self, input: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.model.eval()
            output = self.model(input)

        return output.detach()

    def onnxruntime_backend(self, input: torch.Tensor) -> torch.Tensor:
        sess = ort.InferenceSession(self.model.SerializeToString())
        # Assumptions: 1 input / 1 output
        output_names = [output.name for output in sess.get_outputs()]
        input_names = [input.name for input in sess.get_inputs()]
        # TODO Support gpu
        input_feed_dict = {input_names[0]: input.detach().cpu().numpy()}
        output = sess.run(output_names, input_feed_dict)

        return torch.from_numpy(output[0])

    def npu_backend(self):
        pass
