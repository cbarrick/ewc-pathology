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
usage: main.py [-h] [-n N] [-k N] [-e N] [-p N] [-w N] [-b N] [-c N] [-d]
               [-l LOG_LEVEL] [--name NAME]

Run the EWC experiment.

optional arguments:
  -h, --help            show this help message and exit
  -n N, --data-size N   the number of training samples is a function of N
  -k N, --n-folds N     the number of cross-validation folds
  -e N, --epochs N      the maximum number of epochs per task
  -p N, --patience N    higher patience may help avoid local minima
  -w N, --ewc-strength N
                        the regularization strength of EWC
  -b N, --batch-size N  the batch size
  -c N, --cuda N        use the Nth cuda device
  -d, --dry-run         do a dry run to check for errors
  -l LOG_LEVEL, --log-level LOG_LEVEL
                        set the log level
  --name NAME           sets a name for the experiment
```

The exact command used to run the reported experiment is:

```
$ ./main.py \
    --n-folds=5 \
    --batch-size=1024 \
    --data-size=10000 \
    --epochs=50 \
    --ewc-strength=15
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
