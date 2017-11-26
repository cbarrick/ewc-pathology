import logging
import sys

import torch
import torch.autograd as A
import torch.nn.functional as F


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

    def __init__(self, model, opt, loss, cuda=None):
        '''Create an EWC trainer to train a model on multiple tasks.

        Args:
            model: The model to train.
            opt: The optimizer to step during training.
            loss: The loss function to minimize.
            cuda: The cuda device to use.
        '''
        if cuda is not None:
            model = model.cuda(cuda)

        self.model = model
        self.opt = opt
        self.cuda = cuda

        self._loss = loss
        self._tasks = []

    def variable(self, x, **kwargs):
        '''Cast a Tensor to a Variable on the same cuda device as the model.
        '''
        x = A.Variable(x, **kwargs)
        if self.cuda is not None:
            x = x.cuda(self.cuda)
        return x

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
        x = self.variable(x, volatile=True)
        y = self.variable(y, volatile=True)
        h = self.model(x)
        l = F.log_softmax(h)[:, y]  # log-likelihood of true class
        l.backward()
        grads = (p.grad.data for p in self.params())
        fisher = [(g ** 2).mean(0) for g in grads]
        return fisher

    def consolidate(self, data, alpha=1):
        '''Consolidate the weights given a sample of the current task.

        This adds an EWC regularization term to the loss function.
        '''
        params = [p.data.copy() for p in self.params()]
        fisher = [torch.zeroes_like(p) for p in self.params()]

        n = len(data)
        for x, y in data:
            for i, f in enumerate(self.fisher_information(x, y)):
                fisher[i] += f / n

        task = {
            'params': params,
            'fisher': fisher,
            'alpha': alpha,  # The name 'lambda' is taken by the keyword.
        }

        self._tasks.append(task)

    def loss(self, h, y):
        '''Compute the loss between hypotheses and true label.

        Args:
            h: The hypotheses.
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
            ewc = (l/2 * e for l, e in zip(task['alpha'], ewc))
            j += sum(ewc)
        return j

    def partial_fit(self, x, y):
        '''Performs one step of the optimization.

        Args:
            x: The input batch.
            y: The class labels.
        '''
        self.model.train()  # put the model in train mode, effects dropout layers etc.
        self.opt.zero_grad()  # reset the gradients of the trainable variables.
        x = self.variable(x)
        y = self.variable(y)
        h = self.model(x)
        j = self.loss(h, y)
        j.backward()
        self.opt.step()
        return j.data

    def fit(self, train, validation=[], max_epochs=100, out=sys.stdout):
        '''Fit the model to a task.
        '''
        for epoch in range(max_epochs):

            print(f'epoch {epoch+1} [0%]', end='\r', flush=True, file=out)
            loss_t = 0
            for i, (x, y) in enumerate(train):
                j = self.partial_fit(x, y)
                loss_t += j.sum() / len(train)
                progress = (i+1) / len(train)
                print(f'epoch {epoch+1} [{progress:.2%}]', end='\r', flush=True, file=out)

            loss_v = self.test(validation)

            print(f'epoch {epoch+1}', end='', file=out)
            print(f' [Train loss: {loss_t:8.6f}]', end='', file=out)
            print(f' [Validation loss: {loss_v:8.6f}]', end='', file=out)
            print()

        return loss_t

    def predict(self, x):
        '''Evaluate the model on some input batch.

        Args:
            x: The input batch.
        '''
        self.model.eval()  # put the model in eval mode, effects dropout layers etc.
        x = self.variable(x, volatile=True)  # use volatile input to save memory when not training.
        x = x.cuda(self.cuda) if self.cuda is not None else x
        h = self.model(x)
        return h

    def score(self, x, y, criteria=None):
        '''Score the model against some criteria.

        If the criteria is None, the loss is returned.
        '''
        h = self.predict(x)
        j = self.loss(h, y) if criteris is None else criteria(h, y)
        return j.data

    def test(self, data, criteria=None):
        '''Test the model against some task.
        '''
        n = len(data)
        loss = 0
        for x, y in data:
            j = self.score(x, y, criteria)
            loss += j.sum() / n
        return loss
