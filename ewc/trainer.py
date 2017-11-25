import argparse
from model import  alexnet
import logging
logging.basicConfig(level = logging.DEBUG)

from datasets.nuclei import NucleiLoader
from datasets.epi import EpitheliumLoader
logger=logging.getLogger(__name__)

import torch
torch.manual_seed(1234)

#parser loading arguments
parser = argparse.ArgumentParser()
parser.add_argument("-e", type=int)
args = parser.parse_args()
print(args)
EPOCHS = args.e


class Trainer:

    def __init__(self, classes, batch_epoch = 1):

        self.batch_epoch = batch_epoch
        self.model = alexnet.AlexNet(classes);


    """
    Trains model for specific task alos supports batch based iterations
    name: The name of task to be given useful for logging purposes
    train_x : Torch tensor to be accepted by the model as x arguments
    train_y : Torch tensor to be accepted by the model as y arguments

    """
    def train_on_task(self, name, train_x, train_y):

        logger.info("Training task {}".format(name))
        for i in range(0, self.batch_epoch):
            train_x = train_x.permute(0, 3, 1,2)
            loss = self.model.partial_fit(train_x, train_y)
            logger.debug("loss: {}".format(loss))

        return train_x, train_y

    """
    Trains model for the consolidate phase of the ewc
    name: The name of task to be given useful for logging purposes
    train_x : Torch tensor to be accepted by the model as x arguments
    train_y : Torch tensor to be accepted by the model as y arguments

    """
    def train_on_task_consolidate(self, name, train_x, train_y):

        logger.info("Training task {}".format(name))
        train_x = train_x.permute(0, 3, 1, 2)
        self.model.consolidate(train_x, train_y)


    """
    Trains model for the consolidate phase of the ewc
    name: The name of task to be given useful for logging purposes
    train_x : Torch tensor to be accepted by the model as x arguments
    train_y : Torch tensor to be accepted by the model as y arguments

    """
    def predict(self, test_x):

        test_x = test_x.permute(0, 3, 1, 2)
        return  self.model.predict(test_x)

    """
    Trains model for the  partial fit phase of the  ewc
    name: The name of task to be given useful for logging purposes
    train_x : Torch tensor to be accepted by the model as x arguments
    train_y : Torch tensor to be accepted by the model as y arguments

    """
    def partial_fit(self, data_x1, data_y1):

        first_data=self.train_on_task("MNIST", data_x1, data_y1)
        logger.info("Task is trained ")
        #woc=self.get_accuracy(self.model.predict(first_data[0]),data_y1)
