import numpy as np
import torch
import torch.nn as nn
from torchvision import models


class PseudoLabelGen(nn.Module):
    def __init__(self, args, mode="train"):
        super(PseudoLabelGen, self).__init__()
        self.args = args
        self.visual_extractor = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)

        if self.args.dataset_name == "iu_xray":
            self.linear_classifier = nn.Sequential(
                nn.Dropout(p=0.2), nn.Linear(in_features=2000, out_features=14)
            )
        else:
            self.linear_classifier = nn.Sequential(
                nn.Dropout(p=0.2), nn.Linear(in_features=1000, out_features=14)
            )

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)

    def forward(self, images, mode="train", update_opts={}):
        if self.args.dataset_name == "iu_xray":
            # IU_XRAY Dataset has 2 images per example
            visual_extractor_features_0 = self.visual_extractor(images[:, 0])
            visual_extractor_features_1 = self.visual_extractor(images[:, 1])

            visual_extractor_features = torch.cat(
                (visual_extractor_features_0, visual_extractor_features_1),
                dim=1,
            )

            output = self.linear_classifier(visual_extractor_features)
        else:
            # MIMIC-CXR Dataset has only 1 images per example
            visual_extractor_features = self.visual_extractor(images)
            output = self.linear_classifier(visual_extractor_features)
        return output
