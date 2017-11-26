import logging

import torch
import torch.autograd as A
import torch.nn as N
import torch.nn.functional as F
import torch.optim as O


logger = logging.getLogger(__name__)


class EWCTrainer:
    '''A model trainer that supports multiple tasks through EWC.

    Elastic weight consolidation (EWC) is a method for training a single model
    to perform multiple tasks. The idea is that we first train the model to
    perform a single task, then we "consolidate" the parameters into a new
    regularization term on the loss function before we start training on a new
    task. This regularization allows the model to learn the new task while
    keeping parameters which are important for the first task near to their
    original values. Once we have trained on the second task, we can then
    consolidate the weights again before training on a new task.

    See the paper "Overcoming catastrophic forgetting in neural networks"
    (https://arxiv.org/abs/1612.00796)
    '''

    def __init__(self, model, opt, loss):
        '''Create an EWC trainer to train a model on multiple tasks.

        Args:
            model: The model to train.
            opt: The optimizer to step during training.
            loss: The loss function to minimize.
        '''
        self.model = model
        self.opt = opt
        self._loss = loss
        self._tasks = []

    def params(self):
        '''Get the trainable paramaters from the optimizer.
        '''
        return [p for group in self.opt.param_groups for p in group['params']]

    def fisher_information(self, x, y):
        '''Compute the Fisher information of the trainable parameters.

        Args:
            x: A sample of inputs.
            y: The associated labels.
        '''
        self.model.eval()
        self.opt.zero_grad()
        x = A.Variable(x, volatile=True)
        h = self.model(x)
        l = F.log_softmax(h)[:, y]  # log-likelihood of true class
        l.backward()
        grads = (p.grad.detach().data for p in self.params())
        fisher = [(g ** 2).mean(0) for g in grads]
        return fisher

    def consolidate(self, x, y, lambda=1):
        '''Consolidate the weights of the current task.

        This adds an EWC regularization term to the loss function.

        Args:
            x: A sample of inputs for the task.
            y: The associated labels.
            lambda: The regularization constant for the new EWC term.
        '''
        params = [p.detach().data for p in self.params()]
        fisher = self.fisher_information(x, y)
        task = {
            'params': params,
            'fisher': fisher,
            'lambda': lambda,
        }
        self._tasks.append(task)

    def loss(self, h, y):
        '''Compute the loss with EWC regularization.

        Args:
            h: The hypothesis.
            y: The labels.
        '''
        j = self._loss(h, y)

        # Return the normal loss if there are no consolidated tasks.
        if len(self._tasks) == 0:
            return j

        # Add the ewc regularization for each consolidated task.
        params = self.params()
        for task in self._tasks:
            ewc = ((p - t) ** 2 for t, p in zip(task['params'], params))
            ewc = (f * e for f, e in zip(task['fisher'], ewc))
            ewc = (l/2 * e for l, e in zip(task['lambda'], ewc)
            j += sum(ewc)
        return j

    def predict(self, x):
        '''Evaluate the model on some input batch.

        Args:
            x: The input batch.
        '''
        self.model.eval()  # put the model in eval mode, effects dropout layers etc.
        x = A.Variable(x, volatile=True)  # use volatile input to save memory when not training.
        h = self.model(x)
        return h

    def partial_fit(self, x, y):
        '''Performs one step of the optimization.

        Args:
            x: The input batch.
            y: The class labels.
        '''
        self.model.train()  # put the model in train mode, effects dropout layers etc.
        self.opt.zero_grad()  # reset the gradients of the trainable variables.
        x = A.Variable(x)
        y = A.Variable(y)
        h = self.model(x)
        j = self.loss(h, y)
        j.backward()
        j = j.data[0]
        self.opt.step()
        return j
