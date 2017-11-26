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


def main():
    n_folds = 5

    net = AlexNet(2)
    opt = O.Adam(net.parameters())
    loss = N.CrossEntropyLoss()
    model = EWCTrainer(net, opt, loss)

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
            train, validation, _ = loader.load(f)
            model.fit(train)
            model.consolidate(validation)
            print()
        for task, loader in tasks.items():
            print(f'-------- Scoring {task} --------')
            _, _, test = loader.load(f)
            for metric, criteria in metrics.items():
                z = model.test(test, criteria)
                print(f'{metric}:', z)
            print()


if __name__ == '__main__':
    main()
