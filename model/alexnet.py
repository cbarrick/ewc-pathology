import torch
import torch.nn as N
import torch.optim as O
import torch.autograd as A
import torchvision.models as Models
import logging

logger=logging.getLogger(__name__)

class AlexNet(N.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = N.Sequential(
            N.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            N.ReLU(inplace=True),
            #N.MaxPool2d(kernel_size=3, stride=2),
            N.Conv2d(64, 192, kernel_size=5, padding=2),
            N.ReLU(inplace=True),
            #N.MaxPool2d(kernel_size=3, stride=2),
            N.Conv2d(192, 384, kernel_size=3, padding=1),
            N.ReLU(inplace=True),
            N.Conv2d(384, 256, kernel_size=3, padding=1),
            N.ReLU(inplace=True),
            N.Conv2d(256, 256, kernel_size=3, padding=1),
            N.ReLU(inplace=True),
            #N.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = N.Sequential(
            N.Dropout(),
            N.Linear(256 * 6 * 6, 4096),
            N.ReLU(inplace=True),
            N.Dropout(),
            N.Linear(4096, 4096),
            N.ReLU(inplace=True),
            N.Linear(4096, num_classes),
        )
        logger.debug("CNN Initialised.")
        self.lossfn=N.CrossEntropyLoss()
        self.optimizer=O.Adam(self.parameters())
        logger.debug("Loss function:{} Optimizer:{}".format(self.lossfn,self.optimizer))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

    def partial_fit(self,X,y):
        X=A.Variable(torch.from_numpy(X))
        y=A.Variable(torch.from_numpy(y))
        out=self.forward(X.float())
        loss=self.lossfn(out,y.long())
        logger.info("Loss: {}".format(loss.data[0]))
        loss.backward()
        self.optimizer.step()
