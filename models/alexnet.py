import torch
import torch.nn as N
import torch.nn.functional as F
import torch.optim as O
import torch.autograd as A
import logging

logger=logging.getLogger(__name__)

class AlexNet(N.Module):

    def __init__(self, num_classes=2):
        super(AlexNet, self).__init__()
        self.fisher={}
        self.mean={}
        self.__CONSOLIDATED=0
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
            N.Linear(64, self.num_classes),
            N.Dropout(),
            N.ReLU(inplace=True),
        )
        logger.debug("CNN Initialised.")
        self.lossfn=N.CrossEntropyLoss()
        self.optimizer=O.Adam(self.parameters())
        logger.debug("Loss function:{} Optimizer:{}".format(self.lossfn,self.optimizer))

    def forward(self, x):
        x = self.features(x)
        #print(x)
        x = x.view(x.size(0), 256 * 5 * 5)
        x = self.classifier(x)
        return x

    def partial_fit(self,X,y):
        X=A.Variable(X)
        y=A.Variable(y)
        out=self.forward(X)
        loss=self.lossfn(out,y.long())
        logger.info("Loss: {}".format(loss.data[0]))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.data[0]

    def predict(self,x):
        x=A.Variable(x)
        out=F.softmax(self.forward(x))
        return torch.max(out.data,1)[1].numpy()

    def get_ewc_loss(self,l=15):
        losses=[]
        for name,param in self.named_parameters():
            mean=A.Variable(self.mean[name])
            fisher=A.Variable(self.fisher[name])
            loss=fisher*((param-mean)**2)
            losses.append(loss.sum())
        return (l/2)*sum(losses)

    def consolidate(self,x,y):
        logger.debug("Consolidating weights.")
        x=A.Variable(x)
        y=A.Variable(y).long()
        size=x.size()[0]
        log_liklihood=F.log_softmax(self.forward(x))[range(size),y.data]
        derivatives=A.grad(log_liklihood.sum(),self.parameters())
        for i,(name,param) in enumerate(self.named_parameters()):
            self.fisher[name]=(derivatives[i].data.clone()**2)/size
            self.mean[name]=param.data.clone()
        self.__CONSOLIDATED=1
