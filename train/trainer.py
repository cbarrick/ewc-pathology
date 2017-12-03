import logging
import sys
from pathlib import Path

import numpy as np

import torch
import torch.autograd as A
import torch.nn.functional as F
import torch.utils.data as D


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

    References:
        "Overcoming catastrophic forgetting in neural networks" https://arxiv.org/abs/1612.00796
    '''

    def __init__(self, net, opt, loss, name='model', cuda=None, dry_run=False):
        '''Create an EWC trainer to train a network on multiple tasks.

        Args:
            net (torch.nn.Module):
                The network to train.
                In addition to being a torch `Module`, the network must also
                expose a `reset` method to (re)initialize the weights.
            opt (torch.optim.Optimizer):
                The optimizer to step during training.
            loss (fn: predicted, true -> loss):
                The loss function to minimize.
                The inputs and output are torch `Variable`s.
            name:
                A name for the model.
                This affects logs and file names.
            cuda:
                The cuda device to use.
                The default is device 0 if cuda is available.
                Set to `False` to disable cuda completely.
            dry_run:
                Cut loops short.
        '''
        if cuda is None:
            cuda = 0 if torch.cuda.is_available() else False
        if cuda is not False:
            net = net.cuda(cuda)

        self.net = net
        self.opt = opt
        self._loss = loss
        self.name = name
        self.cuda = cuda
        self.dry_run = dry_run
        self.reset()

        self.path = Path(f'./parameters/{self.name}.torch')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch()

    def reset(self):
        '''Reset the trainer to it's initial state.

        Returns:
            Returns `self` for method chaining.
        '''
        self.ewc = None
        self.net.reset()
        return self

    def save(self, path=None):
        '''Saves the model parameters to disk.

        Returns:
            Returns `self` for method chaining.
        '''
        if path is None:
            path = self.path
        logger.info(f'saving {self.name} to {path}')
        state = self.net.state_dict()
        torch.save(state, str(path))
        return self

    def load(self, path=None):
        '''Loads the model parameters from disk.

        Returns:
            Returns `self` for method chaining.
        '''
        if path is None:
            path = self.path
        logger.info(f'restoring {self.name} from {path}')
        state = torch.load(str(path))
        self.net.load_state_dict(state)
        return self

    def variable(self, x, **kwargs):
        '''Cast `x` to a torch `Variable` on the same cuda device as the network.

        If `x` is already a `Variable`, it is not wrapped.

        Args:
            x:
                The value to cast.
            **kwargs:
                Passed to the `Variable` constructor.
        '''
        if not isinstance(x, A.Variable):
            x = A.Variable(x, **kwargs)
        if self.cuda is not False:
            x = x.cuda(self.cuda, async=True)
        return x

    def tensor(self, x, **kwargs):
        '''Cast `x` to a torch `Tensor` on the same cuda device as the network.

        If `x` is a `Variable`, it is detached from the current graph and
        it's `data` attribute is returned. Otherwise `x` is passed to the
        `Tensor` constructor.

        Args:
            x:
                The value to cast.
            **kwargs:
                Passed to the `Tensor` constructor.
                As of PyTorch 0.2.0, no kwargs are accepted.
        '''
        if isinstance(x, A.Variable):
            x = x.detach().data
        else:
            x = torch.Tensor(x, **kwargs)
        if self.cuda is not False:
            x = x.cuda(self.cuda, async=True)
        return x

    def params(self):
        '''Get the list of trainable paramaters.

        Returns:
            A list of all parameters in the optimizer's `param_groups`.
        '''
        return [p for group in self.opt.param_groups for p in group['params']]

    def fisher_information(self, x, y):
        '''Estimate the Fisher information of the trainable parameters.

        Args:
            x: Some input samples.
            y: Some class labels.

        Returns:
            Returns the fisher information of the trainable parameters.
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
        fisher = [(g ** 2) for g in grads]
        return fisher

    def consolidate(self, data, alpha=1, **kwargs):
        '''Consolidate the weights given a sample of the current task.

        This adds an EWC regularization term to the loss function.

        Args:
            data:
                A dataset of samples from the current task.
                This is typically the validation set.
            alpha:
                The regularization strength for this task.
            **kwargs:
                Forwarded to torch's `DataLoader` class, with more sensible defaults.
        '''
        kwargs.setdefault('pin_memory', self.cuda is not False)
        kwargs['drop_last'] = True
        data = D.DataLoader(data, **kwargs)

        params = [p.clone() for p in self.params()]
        fisher = [torch.zeros(p.size()) for p in self.params()]
        fisher = [self.variable(f) for f in fisher]

        n = len(data)
        for x, y in data:
            for i, f in enumerate(self.fisher_information(x, y)):
                fisher[i] += self.variable(f) / n
            if self.dry_run:
                break

        self.ewc = {
            'params': params,
            'fisher': fisher,
            'alpha': alpha,  # The name 'lambda' is taken by the keyword.
        }

    def loss(self, h, y):
        '''Compute the EWC loss between predictions and their true labels.

        Args:
            h: The predicted labels.
            y: The true labels.

        Returns:
            Returns the base loss plus EWC regularization.
        '''
        j = self._loss(h, y)

        # Return the normal loss if there are no consolidated tasks.
        if self.ewc is None:
            return j

        # Add the ewc regularization for each consolidated task.
        params = self.params()
        a = self.ewc['alpha']
        ewc = ((p - t) ** 2 for t, p in zip(self.ewc['params'], params))
        ewc = (f * e for f, e in zip(self.ewc['fisher'], ewc))
        ewc = (a/2 * e for e in ewc)
        ewc = (e.sum() for e in ewc)
        j += sum(ewc)
        return j

    def partial_fit(self, x, y):
        '''Performs one step of the optimization.

        Args:
            x: The input samples.
            y: The class labels.

        Returns:
            Returns the average loss for this batch.
        '''
        self.net.train()  # put the net in train mode, effects dropout layers etc.
        self.opt.zero_grad()  # reset the gradients of the trainable variables.
        x = self.variable(x)
        y = self.variable(y)
        h = self.net(x)
        j = self.loss(h, y)
        j.backward()
        self.opt.step()
        return j.data.mean()

    def fit(self, train, validation=None, epochs=100, patience=None, **kwargs):
        '''Fit the model to a dataset.

        Args:
            train:
                A dataset to fit.
            validation:
                A dataset to use as validation.
            epochs:
                The maximum number of epochs to spend training.
            patience:
                Stop if the loss does not improve after this many epochs,
                using the validation loss if possible and the train loss otherwise.
                Set to None to disable early stopping.
            **kwargs:
                Forwarded to torch's `DataLoader` class, with more sensible defaults.

        Returns:
            Returns the validation loss if possible,
            otherwise returns the train loss.
        '''
        kwargs.setdefault('shuffle', True)
        kwargs.setdefault('pin_memory', self.cuda is not False)
        train = D.DataLoader(train, **kwargs)

        best_loss = float('inf')
        p = patience or -1
        for epoch in range(epochs):

            # Train
            n = len(train)
            train_loss = 0
            print(f'epoch {epoch+1} [0%]', end='\r', flush=True, file=sys.stderr)
            for i, (x, y) in enumerate(train):
                j = self.partial_fit(x, y)
                train_loss += j / n
                progress = (i+1) / n
                print(f'epoch {epoch+1} [{progress:.2%}]', end='\r', flush=True, file=sys.stderr)
                if self.dry_run:
                    break
            print(f'epoch {epoch+1}', end=' ', flush=True)
            print(f'[Train loss: {train_loss:8.6f}]', end=' ', flush=True)

            # Validate
            if validation:
                val_loss = self.test(validation, **kwargs)
                print(f'[Validation loss: {val_loss:8.6f}]', end=' ', flush=True)
            print(flush=True)

            # Early stopping
            loss = val_loss if validation else train_loss
            if loss < best_loss:
                best_loss = loss
                p = patience or -1
                self.save()
            else:
                p -= 1
            if p == 0:
                self.load()
                break

        return loss

    def predict(self, x):
        '''Predict the classes of some input batch.

        Args:
            x: The input samples.

        Returns:
            A tensor of predicted classes for each row of the input.
        '''
        self.net.eval()  # put the net in eval mode, effects dropout layers etc.
        x = self.variable(x, volatile=True)  # use volatile input to save memory when not training.
        h = self.net(x)
        _, h = h.max(1)
        return h.data

    def score(self, x, y, criteria=None):
        '''Score the model on a batch of inputs and labels.

        Args:
            x: The input samples.
            y: The class labels.
            criteria: The metric to measure; defaults to the mean loss.

        Returns:
            Returns the result of `criteria(true, predicted)`.
        '''
        self.net.eval()

        if criteria is None:
            # Default - the loss has a different signature than a metric
            x = self.variable(x, volatile=True)
            y = self.variable(y, volatile=True)
            h = self.net(x)
            j = self.loss(h, y)
            return j.data.mean()

        try:
            # Fast path - try to use the GPU
            h = self.predict(x)
            y = self.tensor(y)
            return criteria(y, h)

        except (ValueError, TypeError):
            # Slow path - most sklearn metrics cannot handle Tensors
            logger.debug('computing a slow metric')
            h = h.cpu().numpy()
            y = y.cpu().numpy()
            return criteria(y, h)

    def test(self, data, criteria=None, **kwargs):
        '''Score the model on a dataset.

        Args:
            data:
                A dataset to score against.
            criteria:
                The metric to measure.
                Defaults to the mean loss.
            **kwargs:
                Forwarded to torch's `DataLoader` class, with more sensible defaults.

        Returns:
            Returns the result of `criteria(true, predicted)` averaged over all batches.
            The last incomplete batch is dropped by default.
        '''
        kwargs.setdefault('pin_memory', self.cuda is not False)
        kwargs['drop_last'] = True
        data = D.DataLoader(data, **kwargs)
        n = len(data)
        loss = 0
        for x, y in data:
            j = self.score(x, y, criteria)
            loss += j / n
            if self.dry_run:
                break
        return loss
