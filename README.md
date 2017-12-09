# Multitask learning for Digital Pathology

This experiment implements [elastic weight consolidation][ewc] to train a single network to perform two distinct computer vision tasks in Digital Pathology using a single set of parameters.

[ewc]: https://arxiv.org/abs/1612.00796


## Reproducing the experiment

The experiment can be executed by running the `experiments.pathology` module from the root of this repository:

```
python -m experiments.pathology
```

**TODO:** Document the exact command which generates the reported results.

The experiment accepts a number of parameters. Here is the full help output:

```
$ python -m experiments.pathology --help
usage: pathology.py [-n X] [-k X] [-e X] [-l X] [-p X] [-w X] [-b X] [-c X]
                    [-d] [-v] [--seed SEED] [--name NAME] [--help]
                    [TASK [TASK ...]]

Runs an experiment.

Tasks are specified with either a plus (+) or a minus (-) followed by the
name of a dataset. Tasks beginning with a plus fit the model to the dataset,
while tasks beginning with a minus test the model against a dataset.

For example, the default task list of `+nuclei -nuclei` will first fit the
model to the nuclei dataset, then test against against the nuclei dataset.
EWC terms are computed after each fitting and are used for subsequent fits.

Since tasks may begin with a minus (-) you need to separate the task list
from the other arguments by using a double-dash (--). For example:

    python -m experiments.pathology --cuda=0 -- +nuclei -nuclei

Note that the experiment is intended to be executed from the root of the
repository using `python -m`.

Hyper-parameters:
  -n X, --data-size X
  -k X, --folds X
  -e X, --epochs X
  -l X, --learning-rate X
  -p X, --patience X
  -w X, --ewc X

Performance:
  -b X, --batch-size X
  -c X, --cuda X

Debugging:
  -d, --dry-run
  -v, --verbose

Other:
  --seed SEED
  --name NAME
  --help

Positional:
  TASK

Datasets:
  nuclei   A nuclei segmentation dataset
  epi      An epithelium segmentation dataset
```

Protip: To tee the output to a file and keep the pretty progress report in the
terminal, use `python -u`:

```
$ python -u -m experiments.pathology | tee experiment.out
```


## Credits
- Chris Barrick (cbarrick@uga.edu, [@cbarrick](github.com/cbarrick))
- Aditya Shinde (adityas@uga.edu, [@adityashinde1506](github.com/adityashinde1506))
- Prajay Shetty (pjs37741@uga.edu, [@CodeMaster001](github.com/CodeMaster001))
