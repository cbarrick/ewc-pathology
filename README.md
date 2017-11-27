# Multitask learning for Digital Pathology


## Reproducing the experiment

The experiment can be reproduced by executing `main.py` from the root of this repository:

```
$ ./main.py --help
usage: main.py [-h] [-k N] [-b N] [-e N] [-c N] [-d]

Run the EWC experiment.

optional arguments:
  -h, --help            show this help message and exit
  -k N, --n-folds N     The number of cross-validation folds.
  -b N, --batch-size N  The batch size.
  -e N, --epochs N      The maximum number of epochs per task.
  -c N, --cuda N        Use the Nth cuda device.
  -d, --dry-run         Do a dry run to check for errors.
```

The exact command used to run the reported experiment is:

```
$ ./main.py --batch-size=1024 --cuda=0
```

Protip: To tee the output to a file and keep the pretty progress report in the terminal, use `python -u`:

```
$ python -u ./main.py | tee experiment.out
```


## Resources
Proposal:  https://docs.google.com/document/d/1LIX4NpmbNnJfMFjX9pOtjJmBfRRfKbWh5xjthl7jTdI
