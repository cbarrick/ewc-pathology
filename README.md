# Multitask learning for Digital Pathology

This experiment implements [elastic weight consolidation][ewc] to train a single
network to perform two distinct computer vision tasks in Digital Pathology using
a single set of parameters.

[ewc]: https://arxiv.org/abs/1612.00796


## Reproducing the experiment

The experiment can be reproduced by executing `main.py` from the root of this
repository:

```
$ ./main.py --help
usage: main.py [-n N] [-k N] [-e N] [-p N] [-w N] [-b N] [-c N] [-d] [-v]
               [--name NAME] [--help]

Run an experiment.

Hyper-parameters:
  -n N, --data-size N   The number of training samples is a function of N.
  -k N, --folds N       The number of cross-validation folds.
  -e N, --epochs N      The maximum number of epochs per task.
  -p N, --patience N    Higher patience may help avoid local minima.
  -w N, --ewc N         The regularization strength of EWC.

Performance:
  -b N, --batch-size N  The batch size.
  -c N, --cuda N        Use the Nth cuda device.

Debugging:
  -d, --dry-run         Do a dry run to check for errors.
  -v, --verbose         Turn on debug logging.

Other:
  --name NAME           Sets a name for the experiment.
  --help                Show this help message and exit.
```

The exact command used to run the reported experiment is:

```
$ ./main.py \
    --data-size=10000 \
    --folds=5 \
    --epochs=50 \
    --ewc=15 \
    --batch-size=1024 \
    --cuda=0
```

Protip: To tee the output to a file and keep the pretty progress report in the
terminal, use `python -u`:

```
$ python -u ./main.py | tee experiment.out
```


## Credits
- Chris Barrick (cbarrick@uga.edu, [@cbarrick](github.com/cbarrick))
- Aditya Shinde (adityas@uga.edu, [@adityashinde1506](github.com/adityashinde1506))
- Prajay Shetty (pjs37741@uga.edu, [@CodeMaster001](github.com/CodeMaster001))
