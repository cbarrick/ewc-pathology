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
from models import AlexNet
from train import EWCTrainer
from metrics import precision, recall, f_score


logger = logging.getLogger()


def main(**kwargs):
    kwargs.setdefault('data_size', 10000)
    kwargs.setdefault('folds', 5)
    kwargs.setdefault('epochs', 100)
    kwargs.setdefault('patience', 10)
    kwargs.setdefault('ewc', 1)
    kwargs.setdefault('batch_size', 128)
    kwargs.setdefault('cuda', None)
    kwargs.setdefault('dry_run', False)
    kwargs.setdefault('name', 'ewc')
    kwargs.setdefault('log', 'DEBUG')
    kwargs.setdefault('tasks', ['nuclei', 'epi'])
    args = SimpleNamespace(**kwargs)

    logging.basicConfig(
        level=args.log,
        style='{',
        format='[{levelname:.4}][{asctime}][{name}:{lineno}] {msg}',
    )

    net = AlexNet(2)
    opt = O.Adam(net.parameters())
    loss = N.CrossEntropyLoss()
    model = EWCTrainer(net, opt, loss, name=args.name, cuda=args.cuda, dry_run=args.dry_run)

    datasets = {
        'nuclei': NucleiSegmentation(n=args.data_size, k=args.folds),
        'epi': EpitheliumSegmentation(n=args.data_size, k=args.folds),
    }

    metrics = {
        'precision': precision,
        'recall': recall,
        'f-score': f_score,
    }

    data_args = {
        'batch_size': args.batch_size,
        'pin_memory': args.cuda is not False,
    }

    for f in range(args.folds):
        print(f'================================ Fold {f} ================================')
        model.reset()

        for task in args.tasks:
            print(f'-------- Training on {task} --------')
            loader = datasets[task]
            train, validation, _ = loader.load(f)
            model.fit(train, validation, max_epochs=args.epochs, patience=args.patience, **data_args)
            model.consolidate(validation, alpha=args.ewc, **data_args)
            print()

        for task in args.tasks:
            print(f'-------- Scoring {task} --------')
            loader = datasets[task]
            _, _, test = loader.load(f)
            for metric, criteria in metrics.items():
                z = model.test(test, criteria, **data_args)
                print(f'{metric}:', z)
            print()

        if args.dry_run:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run an experiment.',
        add_help=False,
        argument_default=argparse.SUPPRESS,
    )

    parser.add_argument('tasks', metavar='TASK', nargs='*', help='The list of tasks to experiment with.')

    group = parser.add_argument_group('Hyper-parameters')
    group.add_argument('-n', '--data-size', metavar='N', type=int, help='The number of training samples is a function of N.')
    group.add_argument('-k', '--folds', metavar='N', type=int, help='The number of cross-validation folds.')
    group.add_argument('-e', '--epochs', metavar='N', type=int, help='The maximum number of epochs per task.')
    group.add_argument('-p', '--patience', metavar='N', type=int, help='Higher patience may help avoid local minima.')
    group.add_argument('-w', '--ewc', metavar='N', type=float, help='The regularization strength of EWC.')

    group = parser.add_argument_group('Performance')
    group.add_argument('-b', '--batch-size', metavar='N', type=int, help='The batch size.')
    group.add_argument('-c', '--cuda', metavar='N', type=int, help='Use the Nth cuda device.')

    group = parser.add_argument_group('Debugging')
    group.add_argument('-d', '--dry-run', action='store_true', help='Do a dry run to check for errors.')
    group.add_argument('-v', '--verbose', action='store_const', const='DEBUG', help='Turn on debug logging.')

    group = parser.add_argument_group('Other')
    group.add_argument('--name', type=str, help='Sets a name for the experiment.')
    group.add_argument('--help', action='help', help='Show this help message and exit.')

    args = parser.parse_args()
    main(**vars(args))
