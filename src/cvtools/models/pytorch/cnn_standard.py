"""
Base class for CNN models.
"""

# Author: Atif Khurshid
# Created: 2026-10-09
# Modified: None
# Version: 1.0
# Changelog:
#     - 2026-10-09: Initial version.

from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms.v2 as transforms

from .cnn import PyTorchCNNModel


class StandardCNN(PyTorchCNNModel):

    supported_models = ["alexnet", "vgg16", "resnet18", "resnet34", "resnet50"]

    def __init__(
        self,
        name: str,
        n_classes: int,
        pretrained: bool = False,
    ):
        """
        Standard CNN model with optional pretrained weights.

        Parameters
        ----------
        name : str
            Name of the CNN model. Use StandardCNN.supported_models to see the list of supported models.
        n_classes : int
            Number of classes in the classification task.
        pretrained : bool, optional
            If True, use pretrained weights. Default is False.
        """
        if name not in self.supported_models:
            raise ValueError(f"Unsupported model name: {name}. Supported models are: {self.supported_models}")

        super().__init__(classifier=None)

        if pretrained:
            pmean = [0.485, 0.456, 0.406]
            pstd = [0.229, 0.224, 0.225]
        else:
            pmean = [0.5, 0.5, 0.5]
            pstd = [0.5, 0.5, 0.5]

        self.preprocessing = transforms.Normalize(mean = pmean, std = pstd)

        if name == "alexnet":
            if pretrained:
                weights = models.AlexNet_Weights.DEFAULT
            else:
                weights = None
            model = models.alexnet(weights=weights)
            self.features = model.features
        elif name == "vgg16":
            if pretrained:
                weights = models.VGG16_Weights.DEFAULT
            else:
                weights = None
            model = models.vgg16(weights=weights)
            self.features = model.features
        elif "resnet" in name:
            if name == "resnet18":
                if pretrained:
                    weights = models.ResNet18_Weights.DEFAULT
                else:
                    weights = None
                model = models.resnet18(weights=weights)
            elif name == "resnet34":
                if pretrained:
                    weights = models.ResNet34_Weights.DEFAULT
                else:
                    weights = None
                model = models.resnet34(weights=weights)
            elif name == "resnet50":
                if pretrained:
                    weights = models.ResNet50_Weights.DEFAULT
                else:
                    weights = None
                model = models.resnet50(weights=weights)
            self.features = nn.Sequential(OrderedDict(list(model.named_children())[:-2]))
        else:
            raise ValueError(f"Unsupported model name: {name}")

        self.avgpool = model.avgpool

        if "resnet" in name:
            self.classifier = nn.Sequential(OrderedDict(list(model.named_children())[-1:]))
        else:
            self.classifier = model.classifier

        if n_classes != 1000:
            self.classifier[-1] = nn.Linear(self.classifier[-1].in_features, n_classes)


    def forward(self, x):
        x = self.preprocessing(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x
    