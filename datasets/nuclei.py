import re
import tarfile
from pathlib import Path

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import requests

import torch
import torch.utils.data


def get_data(force_download=False):
    '''Download the data and return a Path to the directory.'''
    src = 'http://andrewjanowczyk.com/wp-static/nuclei.tgz'
    arc = Path('./nuclei.tgz')
    dst = Path('./data/nuclei')

    if dst.exists() and not force_download:
        return dst

    print(f'downloading {src}')
    r = requests.get(src, stream=True)
    with arc.open('wb') as fd:
        for chunk in r.iter_content(chunk_size=128):
            fd.write(chunk)
    dst.mkdir(parents=True, exist_ok=True)
    tarfile.open(arc).extractall(dst)
    arc.unlink()
    return dst


def imread(path, mask=False):
    '''Open an image as a numpy array.

    Masks are opened as binary images with shape (N x M).
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
    # estimate location of nuclei where red is low
    red = image[:,:,0]
    mask_b = (red > 0.3).astype('float32')

    # block out areas that likely contain nuclei
    k = np.ones((50, 50))
    mask_b = cv2.erode(mask_b, k)

    return mask_b


def edge_mask(p_mask):
    '''Creates a mask around the outer edges of the positive class.
    '''
    k = np.ones((5,5))
    mask_p = mask_p.astype('uint8')
    edges = cv2.dilate(mask_p, k)
    edges = edges - mask_p
    mask_e = edges.astype('bool')
    return mask_e


def extract_from_mask(image, mask, max_count=None, size=64, random=True):
    '''Sample patches from an image whose centers are not masked.
    '''
    ar = np.require(image) # no copy
    mask = np.array(mask)  # copy
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
    if not max_count:
        max_count = n
    else:
        max_count = min(max_count, n)

    # sample from the image where the mask is True
    if not random:
        idx = np.arange(max_count)
    else:
        idx = np.random.choice(n, max_count, replace=False)
    for i in idx:
        h, k = x[i], y[i]
        patch = ar[h-delta:h+delta, k-delta:k+delta]
        yield patch


def extract_patches(image, mask_p, n, pos_ratio=1, edge_ratio=1, bg_ratio=0.3):
    '''Samples labeled patches from an image given a positive mask.

    The negative class is sampled from an edge mask and a background mask,
    which are generated from the input image and positive mask.
    '''
    # generate the masks
    image = np.require(image)
    mask_p = np.require(mask_p)
    mask_e = edge_mask(mask_p)
    mask_b = background_mask(image)

    # get patches for each mask
    p = list(extract_from_mask(image, mask_p, int(n * pos_ratio), random=True))
    e = list(extract_from_mask(image, mask_e, int(n * edge_ratio), random=True))
    b = list(extract_from_mask(image, mask_b, int(n * bg_ratio), random=True))

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


def create_cv(k=5, n=10000, **kwargs):
    '''Extract a training set of patches taken from all images in a directory.

    The dataset is folded for cross-validation by patient id.

    The file names must follow the format:
        - `{patient_id}_{type}_{image_id}_original.tif` for originals.
        - `{patient_id}_{type}_{image_id}_original.tif` for positive masks.

    Kwargs are passed to `extract_patches`.
    '''
    data_dir = get_data()
    masks = sorted(data_dir.glob('*_mask.png'))
    images = sorted(data_dir.glob('*_original.tif'))

    folds = [{'pos':[], 'neg':[]} for _ in range(k)]

    for i, (img_path, mask_path) in enumerate(zip(images, masks)):
        image = imread(img_path)
        p_mask = imread(mask_path, mask=True)
        meta = get_metadata(img_path)
        pos, neg = extract_patches(image, p_mask, n, **kwargs)
        f = hash(meta['patient']) % k
        folds[f]['pos'].extend(pos)
        folds[f]['neg'].extend(neg)

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
            return self._pos[i], np.float32(1)
        else:
            return self._neg[i], np.float32(0)


class NucleiLoader:
    '''A dataloader for the nuclei segmentation dataset.
    '''
    def __init__(self, k=5, n=10000, **kwargs):
        '''Create a dataloader.

        Args:
            k (int):
                The number of cross-validation folds.
            n (int):
                A parameter to determine the number of
                patches drawn from each source image.

        Kwargs:
            size (int, default=64):
                The size of the image patches.
            pos_ratio (default=1):
                The dataset will contain `n * pos_ratio`
                positive patches per source image.
            edge_ratio (default=1):
                The dataset will contain `n * edge_ratio`
                negative edge patches per source image.
            bg_ratio (default=0.3):
                The dataset will contain `n * bg_ratio`
                negative background patches per source image.
        '''
        folds = create_cv(k, **kwargs)
        self.datasets = [NucleiDataset(f['pos'], f['neg']) for f in folds]

    def load_train(self, fold, **kwargs):
        '''Loads the training set for the given fold.

        The returned value is a torch `DataLoader` that iterates over batches
        of the form `(x, y)`. The kwargs are forwarded to the `DataLoader`.

        Some kwargs defaults have been overridden:
            - `batch_size` defaults to 32.
            - `shuffle` defaults to True.
            - `pin_memory` defaults to True if cuda is avaliable.
        '''
        n = len(self.datasets)
        ds = [self.datasets[f] for f in range(n) if f != fold % n]
        ds = torch.utils.data.ConcatDataset(ds)
        return torch.utils.data.DataLoader(ds, **kwargs)

    def load_test(self, fold, **kwargs):
        '''Loads the test set for the given fold.

        The returned value is a torch `DataLoader` that iterates over batches
        of the form `(x, y)`. The kwargs are forwarded to the `DataLoader`.

        Some kwargs defaults have been overridden:
            - `batch_size` defaults to 32.
            - `shuffle` defaults to True.
            - `pin_memory` defaults to True if cuda is avaliable.
        '''
        n = len(self.datasets)
        ds = self.datasets[fold % n]
        return torch.utils.data.DataLoader(ds, **kwargs)
