import re
from pathlib import Path

import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import torch.utils.data


def imread(path, mask=False):
    '''Open an image as a numpy array.

    Masks are opened as binary image with shape (N x M).
    Otherwise, images are opened as float RGB with shape (N x M x 3).
    '''
    path = str(path)
    if mask:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = img.astype('bool')
    else:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = img[:, :, [2,1,0]] # BGR to RGB
        img = img.astype('float32') / 255
    return img


def background_mask(img):
    '''Create a mask separating the background from the foreground.
    '''
    # nuclei where red is low
    red = img[:,:,0]
    b_mask = (red > 0.3).astype('float32')

    # block out areas that likely contain nuclei
    k = np.ones((50, 50))
    b_mask = cv2.erode(b_mask, k)

    return b_mask


def edge_mask(p_mask):
    '''Creates a mask around the outer edges of the positive class.
    '''
    k = np.ones((3,3))
    p_mask = p_mask.astype('uint8')
    edges = cv2.dilate(p_mask, k)
    edges = edges - p_mask
    return edges.astype('bool')


def sample_from_mask(img, mask, size, max_count=None, random=False):
    '''Sample patches from an image whose centers are not masked.
    '''
    img = np.require(img) # no copy
    mask = np.array(mask) # copy
    width, height = mask.shape
    delta = size//2

    # mask off the edges where we can't form a complete patch
    mask[:delta, :] = 0
    mask[:, :delta] = 0
    mask[-delta:, :] = 0
    mask[:, -delta:] = 0

    # find the coords where the mask is True
    x, y = np.where(mask)
    n = len(x)

    # the max_count is at most the number of coords
    if max_count is None:
        max_count = n
    else:
        max_count = min(max_count, n)

    # randomly sample from the image where the mask is True
    if not random:
        idx = np.arange(max_count)
    else:
        idx = np.random.choice(n, max_count, replace=False)
    for i in idx:
        h, k = x[i], y[i]
        patch = img[h-delta:h+delta, k-delta:k+delta]
        yield patch


def sample_labeled(img, p_mask, size=64, edge_ratio=1, bg_ratio=0.3):
    '''Samples labeled patches from an image given a positive mask.

    The negative class is sampled from an edge mask and a background mask,
    which are generated from the input image and positive mask.
    '''
    # generate the masks
    img = np.require(img)
    p_mask = np.require(p_mask)
    e_mask = edge_mask(p_mask)
    b_mask = background_mask(img)

    # get patches for each mask
    n = e_mask.sum()
    p = list(sample_from_mask(img, p_mask, size, n, random=True))
    e = list(sample_from_mask(img, e_mask, size, int(n * edge_ratio), random=True))
    b = list(sample_from_mask(img, b_mask, size, int(n * bg_ratio), random=True))

    # separate into positive and negative classes
    pos = p
    neg = e+b

    # if the classes are imbalanced, throw away some extras
    if len(neg) < len(pos):
        pos = pos[:len(neg)]

    return pos, neg


def get_metadata(image_path):
    '''Extracts the metadata from a file name.
    '''
    image_path = str(image_path)
    match = re.search('([0-9]+)_([0-9]+)_([0-9a-f]+)_', image_path)
    return {
        'id': match[3],
        'type': match[2],
        'patient': match[1],
        'path': image_path,
    }


def create_cv(root, n_folds=5, **kwargs):
    '''Extract a training set of patches taken from all images in a directory.

    The dataset is folded for cross-validation by patient id.

    The file names must follow the format:
        - `{patient_id}_{type}_{image_id}_original.tif` for originals.
        - `{patient_id}_{type}_{image_id}_original.tif` for positive masks.

    Kwargs are passed to `sample_labeled`.
    '''
    print('creating patches', end=' ', flush=True)

    # enumerate the source files
    root = Path(root)
    originals = sorted(root.glob('*_original.tif'))
    masks = sorted(root.glob('*_mask.png'))

    folds = [{'pos':[], 'neg':[]} for _ in range(n_folds)]

    # for every image, mask pair
    for i, (img_path, mask_path) in enumerate(zip(originals, masks)):
        print(end='.', flush=True)
        image = imread(img_path)
        mask = imread(mask_path, mask=True)
        meta = get_metadata(img_path)

        # assign a fold based on patient id
        f = hash(meta['patient']) % n_folds
        fold = folds[f]

        # extract the patches
        pos, neg = sample_labeled(image, mask, **kwargs)
        fold['pos'].extend(pos)
        fold['neg'].extend(neg)
    print(' DONE')
    return folds


class NucleiDataset(torch.utils.data.Dataset):
    '''A torch `Dataset` that combines positive and negative samples.
    '''
    def __init__(self, pos, neg):
        self._pos = pos
        self._neg = neg
        self._split = len(pos)

    def __len__(self):
        return len(self._pos) + len(self._neg)

    def __getitem__(self, i):
        i -= self._split
        if i < 0:
            return self._pos[i], 1
        else:
            return self._neg[i], 0


class NucleiSegmentation:
    '''A dataloader for the nuclei segmentation dataset.
    '''

    def __init__(self, root='./nuclei', n_folds=5):
        '''Create a dataloader.

        Args:
            root (path): The root directory.
            n_folds (int): The number of cross-validation folds.
        '''
        root = Path(root)
        originals = sorted(root.glob('*_original.tif'))
        masks = sorted(root.glob('*_mask.png'))
        folds = create_cv(root, n_folds)
        self.datasets = [NucleiDataset(f['pos'], f['neg']) for f in folds]
        self._n_folds = n_folds

    def load_train(self, fold, **kwargs):
        '''Loads the training set for the given fold.

        The returned value is a torch `DataLoader` that iterates over batches
        of (patch, label) pairs. The kwargs are forwarded to `DataLoader`.

        Some kwargs defaults have been overridden:
            - `batch_size` defaults to 32.
            - `shuffle` defaults to True.
            - `pin_memory` defaults to True if cuda is avaliable.
        '''
        assert 0 <= fold
        assert fold < self._n_folds
        kwargs.setdefault('batch_size', 32)
        kwargs.setdefault('shuffle', True)
        kwargs.setdefault('pin_memory', torch.cuda.is_available())

        ds = self.datasets[:fold] + self.datasets[fold+1:]
        ds = torch.utils.data.ConcatDataset(ds)
        return torch.utils.data.DataLoader(ds, **kwargs)

    def load_test(self, fold, **kwargs):
        '''Loads the test set for the given fold.

        The returned value is a torch `DataLoader` that iterates over batches
        of (patch, label) pairs. The kwargs are forwarded to `DataLoader`.

        Some kwargs defaults have been overridden:
            - `batch_size` defaults to 32.
            - `shuffle` defaults to True.
            - `pin_memory` defaults to True if cuda is avaliable.
        '''
        assert 0 <= fold
        assert fold < self._n_folds
        kwargs.setdefault('batch_size', 32)
        kwargs.setdefault('shuffle', True)
        kwargs.setdefault('pin_memory', torch.cuda.is_available())

        ds = self.datasets[fold]
        return torch.utils.data.DataLoader(ds, **kwargs)
