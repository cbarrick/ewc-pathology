#!/usr/bin/env python3
import argparse
import logging

import numpy as np
import sklearn.metrics

import torch
import torch.autograd as A
import torch.nn as N
import torch.nn.functional as F
import torch.optim as O

from datasets import NucleiLoader
from datasets import EpitheliumLoader
from train import EWCTrainer
from models import AlexNet


logger = logging.getLogger(__name__)


def main(n_folds=5, batch_size=64, epochs=100, cuda=None):
    net = AlexNet(2)
    opt = O.Adam(net.parameters())
    loss = N.CrossEntropyLoss()
    model = EWCTrainer(net, opt, loss, cuda)

    tasks = {
        'nuclei': NucleiLoader(k=n_folds),
        'epithelium': EpitheliumLoader(k=n_folds),
    }

    metrics = {
        'f-measure': sklearn.metrics.f1_score,
        'precision': sklearn.metrics.precision_score,
        'recal': sklearn.metrics.recall_score,
        'log-loss': sklearn.metrics.log_loss,
    }

    for f in range(n_folds):
        print(f'================ Fold {f} ================')
        for task, loader in tasks.items():
            print(f'-------- Training on {task} --------')
            train, validation, _ = loader.load(f, batch_size=batch_size)
            model.fit(train, validation, max_epochs=epochs)
            model.consolidate(validation)
        for task, loader in tasks.items():
            print(f'-------- Scoring {task} --------')
            _, _, test = loader.load(f, batch_size=batch_size)
            for metric, criteria in metrics.items():
                z = model.test(test, criteria)
                print(f'{metric}:', z)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the EWC experiment.')
    parser.add_argument('-k', '--n_folds', metavar='N', type=int, default=5, help='The number of cross-validation folds.')
    parser.add_argument('-b', '--batch_size', metavar='N', type=int, default=64, help='The batch size.')
    parser.add_argument('-e', '--epochs', metavar='N', type=int, default=100, help='The maximum number of epochs per task.')
    parser.add_argument('-c', '--cuda', metavar='N', type=int, default=None, help='Use the Nth cuda device.')
    args = parser.parse_args()
    main(**vars(args))
