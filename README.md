# Multitask learning for Digital Pathology

This experiment implements [elastic weight consolidation][ewc] to train a single network to perform two distinct computer vision tasks in Digital Pathology using a single set of parameters.

[ewc]: https://arxiv.org/abs/1612.00796


## Reproducing the experiment

The experiment can be reproduced by executing `main.py` from the root of this repository:

```
$ ./main.py --help
usage: main.py [-h] [-k N] [-b N] [-e N] [-c N] [-d]

Run the EWC experiment.

optional arguments:
  -h, --help            show this help message and exit
  -k N, --n-folds N     the number of cross-validation folds
  -b N, --batch-size N  the batch size
  -e N, --epochs N      the maximum number of epochs per task
  -c N, --cuda N        use the Nth cuda device
  -d, --dry-run         do a dry run to check for errors
```

The exact command used to run the reported experiment is:

```
$ ./main.py --n-folds=5 --batch-size=1024 --cuda=0
```

Protip: To tee the output to a file and keep the pretty progress report in the terminal, use `python -u`:

```
$ python -u ./main.py | tee experiment.out
```


## Credits
- Chris Barrick (cbarrick@uga.edu, @cbarrick)
- Aditya Shinde (adityas@uga.edu, @adityashinde1506)
- Prajay Shetty (pjs37741@uga.edu, @CodeMaster001)
