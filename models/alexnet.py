import logging

import torch
import torch.nn as N


logger = logging.getLogger(__name__)


class AlexNet(N.Module):
    '''An AlexNet-like model based on the CIFAR-10 variant of AlexNet in Caffe.
    '''

    def __init__(self, num_classes=2):
        super().__init__()

        self.features = N.Sequential(
            N.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            N.MaxPool2d(kernel_size=3, stride=2, padding=1),
            N.ReLU(inplace=True),
            N.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            N.ReLU(inplace=True),
            N.AvgPool2d(kernel_size=3, stride=2, padding=1),
            N.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            N.ReLU(inplace=True),
            N.AvgPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.classifier = N.Sequential(
            N.Linear(64*8*8, 64),
            N.Dropout(),
            N.ReLU(inplace=True),
            N.Linear(64, num_classes),
            N.Dropout(),
            N.ReLU(inplace=True),
        )

        for m in self.modules():
            if isinstance(m, (N.Conv2d, N.Linear)):
                N.init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
