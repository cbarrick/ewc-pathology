import argparse
#sys.path.append("/home/adityas/Projects/DataScience2/model/")

import testnet
import logging
logging.basicConfig(level=logging.DEBUG)

from keras.datasets import mnist,fashion_mnist
import numpy

logger=logging.getLogger(__name__)

parser=argparse.ArgumentParser()
parser.add_argument("--ewc",action='store_true')
parser.add_argument("-e",type=int)
args=parser.parse_args()
print(args)
EPOCHS=args.e

if args.ewc:
    logger.critical("Running with EWC")
else:
    logger.critical("Running without EWC (Vanilla loss)")
    

import torch
torch.manual_seed(1234)

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

(mnist_train_x,mnist_train_y),(mnist_test_x,mnist_test_y)=mnist.load_data()
(fashion_train_x,fashion_train_y),(fashion_test_x,fashion_test_y)=fashion_mnist.load_data()

sample_data=mnist_train_x[:5000],mnist_train_y[:5000]

def get_accuracy(output,label):
    total=output.shape[0]
    correct=output[output==label].shape[0]
    accuracy=correct/total
    logger.info("Accuracy: {}".format(accuracy))
    return accuracy

def train_on_task(name,batch_generator,EPOCHS=EPOCHS):
    logger.info("Training task {}".format(name))
    for i in range(EPOCHS):
        train_x,train_y=next(batch_generator)
        #train_x=numpy.repeat(train_x[:,:,:,numpy.newaxis],3,axis=3)
        train_x=train_x[:,:,:,numpy.newaxis]
        train_x=numpy.transpose(train_x,(0,3,1,2))
        model.partial_fit(train_x,train_y)
        if i==EPOCHS-1:
            logger.info("Accuracy on {}: {}".format(name,get_accuracy(model.predict(train_x),train_y)))
    return train_x,train_y

model=testnet.AlexNet(10)
mnist_data=train_on_task("MNIST",batchify(mnist_train_x,mnist_train_y))
if args.ewc:
    model.consolidate(numpy.transpose(sample_data[0][:,:,:,numpy.newaxis],(0,3,1,2)),sample_data[1])
fashion_data=train_on_task("FASHION",batchify(fashion_train_x,fashion_train_y))
logger.info("Task B is trained. Now computing accuracy for task A to check forgetting.")
woc=get_accuracy(model.predict(mnist_data[0]),mnist_data[1])
