# Models

## Implementation with partial_fit method. 

The partial fit method will run one iteration of the training loop. For input, it takes X in the format (batch_size,channels,height,width) and y in the format (labels,)

**!!!** For this model, I have commented out the max pool layers since i was testing on the MNIST dataset. MaxPool shrunk the inputs to 0. If our images are large enough, we may turn max pooling back on.

Here is an example of the overall training script which will have to written on the outside.

```python
import sys
sys.path.append("/home/adityas/Projects/DataScience2/model/")
 
import alexnet
import logging
logging.basicConfig(level=logging.DEBUG)

from keras.datasets import mnist
import numpy

EPOCHS=1000

def batchify(X,y,batch_size=1000):
    i=0
    while 1:
        batch_X,batch_y=X[i:i+batch_size],y[i:i+batch_size]
        if batch_X.shape[0]!=batch_size:
            i=0
            difference=batch_size-batch_X.shape[0]
            batch_X=numpy.vstack((batch_X,X[i:i+difference]))
            batch_y=numpy.hstack((batch_y,y[i:i+difference]))
        yield batch_X,batch_y
        i+=batch_size

(train_x,train_y),(test_x,test_y)=mnist.load_data()
model=alexnet.AlexNet(num_classes=10)
batch_generator=batchify(train_x,train_y)

for i in range(EPOCHS):
    train_x,train_y=next(batch_generator)
    #Convert input in the give format (batch,channel,H,W)
    train_x=numpy.repeat(train_x[:,:,:,numpy.newaxis],3,axis=3)
    train_x=numpy.transpose(train_x,(0,3,1,2))
    model.partial_fit(train_x,train_y)
```

## Updates

### EWC now works!

The AlexNet model now has a `consolidate(self,x,y)` method for consolidating weights after training on a task.

Now the computation of the Fisher needs samples from the dataset. So we will have to store a few samples from every task during training. and then call `consolidate(taskA_sample_x,taskA_sample_y)` to consolidate the weights for task A before starting training on task B.

If have written a small demo network (`testnet.py`) and a script (`demo.py`) to demonstrate the effect of training with and without EWC.

You'll need an up to date version of keras to run the demo cos I am using keras datasets. So just run `conda install -c conda_forge keras` to update.

To train without EWC, run
`python3 demo.py -e 50`
And notice how accuracy of task A drops significantly after training for task B

Now run, `python3 demo.py -e 50 --ewc`
And now compare the accuracy of task A at the end. 

When we use this, we will have to call `consolidate(self,x,y)` after every task so that the network prepares the fishers for EWC loss for the next task.
Like this
```python
model=testnet.AlexNet(10)
mnist_data=train_on_task("MNIST",batchify(mnist_train_x,mnist_train_y))
model.consolidate(numpy.transpose(taskA_sample_x,taskA_sample_y))
fashion_data=train_on_task("FASHION",batchify(fashion_train_x,fashion_train_y))
``` 