#!/usr/bin/env python3
import argparse
import logging
from types import SimpleNamespace

import numpy as np
import sklearn.metrics

import torch
import torch.autograd as A
import torch.nn as N
import torch.nn.functional as F
import torch.optim as O

from datasets import NucleiSegmentation
from datasets import EpitheliumSegmentation
from train import EWCTrainer
from models import AlexNet


logger = logging.getLogger()


def main(**kwargs):
    kwargs.setdefault('data_size', 10000)
    kwargs.setdefault('n_folds', 5)
    kwargs.setdefault('epochs', 100)
    kwargs.setdefault('patience', 10)
    kwargs.setdefault('ewc_strength', 1)
    kwargs.setdefault('batch_size', 128)
    kwargs.setdefault('cuda', None)
    kwargs.setdefault('dry_run', False)
    kwargs.setdefault('name', 'ewc')
    kwargs.setdefault('log_level', 'DEBUG')
    args = SimpleNamespace(**kwargs)

    logging.basicConfig(
        level=args.log_level,
        style='{',
        format='[{levelname:.4}][{asctime}][{name}:{lineno}] {msg}',
    )

    net = AlexNet(2)
    opt = O.Adam(net.parameters())
    loss = N.CrossEntropyLoss()
    model = EWCTrainer(net, opt, loss, name=args.name, cuda=args.cuda, dry_run=args.dry_run)

    tasks = {
        'nuclei': NucleiSegmentation(n=args.data_size, k=args.n_folds),
        'epithelium': EpitheliumSegmentation(n=args.data_size, k=args.n_folds),
    }

    metrics = {
        'f-measure': sklearn.metrics.f1_score,
        'precision': sklearn.metrics.precision_score,
        'recal': sklearn.metrics.recall_score,
        'log-loss': sklearn.metrics.log_loss,
    }

    data_args = {
        'batch_size': args.batch_size,
        'pin_memory': args.cuda is not False,
    }

    for f in range(args.n_folds):
        print(f'================================ Fold {f} ================================')
        model.reset()

        for task, loader in tasks.items():
            print(f'-------- Training on {task} --------')
            train, validation, _ = loader.load(f)
            model.fit(train, validation, max_epochs=args.epochs, patience=args.patience, **data_args)
            model.consolidate(validation, alpha=args.ewc_strength, **data_args)
            print()

        for task, loader in tasks.items():
            print(f'-------- Scoring {task} --------')
            _, _, test = loader.load(f)
            for metric, criteria in metrics.items():
                z = model.test(test, criteria, **data_args)
                print(f'{metric}:', z)
            print()

        if args.dry_run:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the EWC experiment.')
    parser.add_argument('-n', '--data-size', metavar='N', type=int, help='the number of training samples is a function of N')
    parser.add_argument('-k', '--n-folds', metavar='N', type=int, help='the number of cross-validation folds')
    parser.add_argument('-e', '--epochs', metavar='N', type=int, help='the maximum number of epochs per task')
    parser.add_argument('-p', '--patience', metavar='N', type=int, help='higher patience may help avoid local minima')
    parser.add_argument('-w', '--ewc-strength', metavar='N', type=float, help='the regularization strength of EWC')
    parser.add_argument('-b', '--batch-size', metavar='N', type=int, help='the batch size')
    parser.add_argument('-c', '--cuda', metavar='N', type=int, help='use the Nth cuda device')
    parser.add_argument('-d', '--dry-run', action='store_true', help='do a dry run to check for errors')
    parser.add_argument('-l', '--log-level', help='set the log level')
    parser.add_argument('--name', type=str, help='sets a name for the experiment')
    args = parser.parse_args()
    main(**vars(args))
