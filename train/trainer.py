import logging
import sys

import torch
import torch.autograd as A
import torch.nn.functional as F


logger = logging.getLogger(__name__)


class EWCTrainer:
    '''A network trainer that supports multiple tasks through EWC.

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

    def __init__(self, net, opt, loss, cuda=None, dry_run=False):
        '''Create an EWC trainer to train a network on multiple tasks.

        Args:
            net: The network to train.
            opt: The optimizer to step during training.
            loss: The loss function to minimize.
            cuda: The cuda device to use.
        '''
        if cuda is not None:
            net = net.cuda(cuda)

        self.net = net
        self.opt = opt
        self._loss = loss
        self.cuda = cuda
        self.dry_run = dry_run

        self.reset()

    def reset(self):
        '''Reset the trainer to it's initial state.
        '''
        self._tasks = []
        self.net.reset()
        return self

    def variable(self, x, **kwargs):
        '''Cast a tensor to a `Variable` on the same cuda device as the network.

        If the input is already a `Variable`, it is not wrapped.

        Args:
            x: The tensor to wrap.
            **kwargs: Passed to the `Variable` constructor.
        '''
        if not isinstance(x, A.Variable):
            x = A.Variable(x, **kwargs)
        if self.cuda is not None:
            x = x.cuda(self.cuda, async=True)
        return x

    def params(self):
        '''Get the list of trainable paramaters.

        Returns:
            A list of all parameters in the optimizer's `param_groups`.
        '''
        return [p for group in self.opt.param_groups for p in group['params']]

    def fisher_information(self, x, y):
        '''Compute the Fisher information of the trainable parameters.

        Args:
            x: A batch of inputs.
            y: A batch of labels.

        Returns:
            Returns the fisher information of the trainable parameter.
            The values are arranged similarly to `EWCTrainer.params()`.
        '''
        self.net.eval()
        self.opt.zero_grad()
        x = self.variable(x)
        y = self.variable(y)
        h = self.net(x)
        l = F.log_softmax(h)[range(y.size(0)), y.data]  # log-likelihood of true class
        l = l.mean()
        l.backward()
        grads = (p.grad.data for p in self.params())
        fisher = [(g ** 2).mean(0) for g in grads]
        return fisher

    def consolidate(self, data, alpha=1):
        '''Consolidate the weights given a sample of the current task.

        This adds an EWC regularization term to the loss function.

        Args:
            data: A `torch.DataLoader` of samples from the current task.
            alpha: The regularization strength for this task.
        '''
        params = [p.clone() for p in self.params()]
        fisher = [torch.zeros(p.size()) for p in self.params()]
        fisher = [self.variable(f) for f in fisher]

        n = len(data.dataset)
        for x, y in data:
            for i, f in enumerate(self.fisher_information(x, y)):
                fisher[i] += self.variable(f) / n
            if self.dry_run:
                break

        task = {
            'params': params,
            'fisher': fisher,
            'alpha': alpha,  # The name 'lambda' is taken by the keyword.
        }

        self._tasks.append(task)

    def loss(self, h, y):
        '''Compute the EWC loss between hypotheses and true label.

        Args:
            h: A batch of hypotheses.
            y: A batch of true labels.

        Returns:
            Returns the base loss function plus
            EWC regularization terms for each task.
        '''
        j = self._loss(h, y)

        # Return the normal loss if there are no consolidated tasks.
        if len(self._tasks) == 0:
            return j

        # Add the ewc regularization for each consolidated task.
        params = self.params()
        for task in self._tasks:
            a = task['alpha']
            ewc = ((p - t) ** 2 for t, p in zip(task['params'], params))
            ewc = (f * e for f, e in zip(task['fisher'], ewc))
            ewc = (a/2 * e for e in ewc)
            ewc = (e.sum() for e in ewc)
            j += sum(ewc)
        return j

    def partial_fit(self, x, y):
        '''Performs one step of the optimization.

        Args:
            x: The input batch.
            y: The class labels.

        Returns:
            Returns the sum of losses for this batch.
        '''
        self.net.train()  # put the net in train mode, effects dropout layers etc.
        self.opt.zero_grad()  # reset the gradients of the trainable variables.
        x = self.variable(x)
        y = self.variable(y)
        h = self.net(x)
        j = self.loss(h, y)
        j.backward()
        self.opt.step()
        return j.data.sum()

    def fit(self, train, validation=None, max_epochs=100, patience=5):
        '''Fit the model to a dataset.

        Args:
            train: A `torch.DataLoader` to fit.
            validation: A `torch.DataLoader` to use as the validation set.
            max_epochs: The maximum number of epochs to spend training.
            patience: Stop if the validation loss does not improve after this many epochs.

        Returns:
            Returns the validation loss.
            Returns train loss if no validation set is given.
        '''
        best_loss = float('inf')
        p = patience
        for epoch in range(max_epochs):

            # Train
            train_loss = 0
            print(f'epoch {epoch+1} [0%]', end='\r', flush=True, file=sys.stderr)
            for i, (x, y) in enumerate(train):
                j = self.partial_fit(x, y)
                train_loss += j / len(train.dataset)
                progress = (i+1) / len(train)
                print(f'epoch {epoch+1} [{progress:.2%}]', end='\r', flush=True, file=sys.stderr)
                if self.dry_run:
                    break
            print(f'epoch {epoch+1} [Train loss: {train_loss:8.6f}]', end='')

            # Validate
            if validation:
                val_loss = self.test(validation)
                print(f' [Validation loss: {val_loss:8.6f}]', end='')
            print(flush=True)

            # Convergence test
            loss = val_loss if validation else train_loss
            if loss < best_loss:
                best_loss = loss
                p = patience
            else:
                p -= 1
                if p == 0:
                    break

        return loss

    def predict(self, x):
        '''Predict the classes of some input batch.

        Args:
            x: The input batch.

        Returns:
            A tensor of predicted classes for each row of the input.
        '''
        self.net.eval()  # put the net in eval mode, effects dropout layers etc.
        x = self.variable(x, volatile=True)  # use volatile input to save memory when not training.
        h = self.net(x)
        _, h = h.max(1)
        return h

    def score(self, x, y, criteria=None):
        '''Score the model on a batch of inputs and labels.

        Args:
            x: The input batch.
            y: The targets.
            criteria: The metric to measure; defaults to the loss.

        Returns:
            Returns the result of `criteria(true, predicted)`
        '''
        self.net.eval()
        x = self.variable(x, volatile=True)
        y = self.variable(y, volatile=True)

        if criteria is None:
            h = self.net(x)
            j = self.loss(h, y)
            j = j.data.sum()
        else:
            h = self.predict(x)
            h = h.data.cpu().numpy()
            y = y.data.cpu().numpy()
            j = criteria(y, h, labels=[0,1])

        return j

    def test(self, data, criteria=None):
        '''Score the model on a dataset.

        Args:
            data: A `torch.DataLoader` to score against.
            criteria: The metric to measure; defaults to the loss.
        '''
        n = len(data.dataset)
        loss = 0
        for x, y in data:
            j = self.score(x, y, criteria)
            loss += j / n
            if self.dry_run:
                break
        return loss
