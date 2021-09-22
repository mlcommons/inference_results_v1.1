from typing import List, Tuple

import torch
from torch import nn

from models.ssd_mobilenet_v1 import SSD, MobileNetV1Base, Block
from models.anchor_generator import create_ssd_anchors
from models.utils import decode_boxes


class PredictionHead(nn.Module):
    def __init__(self, in_channels, num_classes, num_anchors):
        super(PredictionHead, self).__init__()
        self.classification = nn.Conv2d(in_channels, num_classes * num_anchors, kernel_size=1)
        self.regression = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1)

        self.num_classes = num_classes
        self.num_anchors = num_anchors

    def forward(self, x):
        class_logits = self.classification(x)
        box_regression = self.regression(x)

        return class_logits, box_regression


class MLCommons_SSDMobileNetV1(SSD):
    num_classes = 91
    input_size = [300, 300]
    # feature map shapes for fixed input_size [300, 300]
    # https://github.com/mlcommons/inference/blob/b62030adda3293904e3d0e756a2218faf166fc45/vision/classification_and_detection/python/models/ssd_mobilenet_v1.py#L155
    feature_map_shapes = [(19, 19), (10, 10), (5, 5), (3, 3), (2, 2), (1, 1)]

    def __init__(self):
        backbone = MobileNetV1Base()
        extras = nn.ModuleList(
            [
                Block(1024, 256, 512),
                Block(512, 128, 256),
                Block(256, 128, 256),
                Block(256, 64, 128),
            ]
        )

        predictors = nn.ModuleList(
            [
                PredictionHead(in_channels, self.num_classes, num_anchors)
                for in_channels, num_anchors in zip(
                    (512, 1024, 512, 256, 256, 128), (3, 6, 6, 6, 6, 6)
                )
            ]
        )
        super(MLCommons_SSDMobileNetV1, self).__init__(backbone, predictors, extras)
        self._feature_map_shapes = self.feature_map_shapes
        self.priors = self.anchor_generation(self.feature_map_shapes)

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores, boxes = self.model(images)

        return scores, boxes

    def model(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # https://github.com/mlcommons/inference/blob/b62030adda3293904e3d0e756a2218faf166fc45/vision/classification_and_detection/python/models/ssd_mobilenet_v1.py#L137-L153
        feature_maps = self.backbone(x)

        out = feature_maps[-1]
        for module in self.extras:
            out = module(out)
            feature_maps.append(out)

        results = []
        for feature, module in zip(feature_maps, self.predictors):
            results.append(module(feature))

        class_logits, box_regression = list(zip(*results))

        return class_logits, box_regression

    @staticmethod
    def anchor_generation(feature_map_shapes: List[Tuple[int, int]]) -> torch.Tensor:
        # https://github.com/mlcommons/inference/blob/b62030adda3293904e3d0e756a2218faf166fc45/vision/classification_and_detection/python/models/ssd_mobilenet_v1.py#L158-L159
        priors = create_ssd_anchors()._generate(feature_map_shapes)
        priors = torch.cat(priors, dim=0)

        return priors

    def preprocess(self):
        raise NotImplemented

    def postprocess(self, class_logits, box_regression):
        bs = class_logits[0].shape[0]
        class_logits = [
            logit.permute(0, 2, 3, 1).reshape(bs, -1, self.num_classes) for logit in class_logits
        ]
        box_regression = [
            regress.permute(0, 2, 3, 1).reshape(bs, -1, 4) for regress in box_regression
        ]

        class_logits = torch.cat(class_logits, 1)
        box_regression = torch.cat(box_regression, 1)

        scores = torch.sigmoid(class_logits)
        box_regression = box_regression.squeeze(0)

        # https://github.com/mlcommons/inference/blob/b62030adda3293904e3d0e756a2218faf166fc45/vision/classification_and_detection/python/models/ssd_mobilenet_v1.py#L164-L166
        if box_regression.dim() == 2:
            box_regression = box_regression[None]
        # decode box_regression into bounding box
        # https://github.com/mlcommons/inference/blob/b62030adda3293904e3d0e756a2218faf166fc45/vision/classification_and_detection/python/models/ssd_mobilenet_v1.py#L155-L167
        boxes = decode_boxes(box_regression, self.priors, self.coder_weights)
        # add a batch dimension

        # sort and nms
        # https://github.com/mlcommons/inference/blob/b62030adda3293904e3d0e756a2218faf166fc45/vision/classification_and_detection/python/models/ssd_mobilenet_v1.py#L178-L185
        list_boxes = []
        list_labels = []
        list_scores = []
        for b in range(len(scores)):
            bboxes, blabels, bscores = self.filter_results(scores[b], boxes[b])
            list_boxes.append(bboxes)
            list_labels.append(blabels.long())
            list_scores.append(bscores)
        # boxes = self.rescale_boxes(boxes, height, width)
        return [list_boxes, list_labels, list_scores]
