# DS2-Term-Project
Data Science II Term Project


## Getting Started

### Downloading and using the datasets
All of the datasets can be downloaded [here][datasets].

Classes for loading the data can be found in the `datasets` package. The resulting data access objects expose two methods, `load_train(i)` and `load_test(i)` where `i` is the index of one of the folds. The value returned is an iterator over tuples `(x, y)` where `x` is a batch of image patches and `y` are the corresponding labels.

#### Example
```python
from datasets import NucleiLoader

data = datasets.NucleiLoader()
model = MySuperAwesomeClassifier()

for epoch in range(100):
    for x, y in data.load_train(0):
        model.partial_fit(x, y)
```

[datasets]: http://www.andrewjanowczyk.com/deep-learning/


## Resources
Proposal:  https://docs.google.com/document/d/1LIX4NpmbNnJfMFjX9pOtjJmBfRRfKbWh5xjthl7jTdI
