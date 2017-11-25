import argparse
#sys.path.append("/home/adityas/Projects/DataScience2/model/")

from model import  alexnet
import logging
logging.basicConfig(level=logging.DEBUG)

from keras.datasets import mnist
import numpy
from scores_individual import Score

from datasets.nuclei import NucleiLoader
from datasets.epi import EpitheliumLoader 
from EWCTrainer import EWCTrainer
logger=logging.getLogger(__name__)

parser=argparse.ArgumentParser()
parser.add_argument("-e",type=int)
args=parser.parse_args()
print(args)
EPOCHS=args.e


data_nuclei = NucleiLoader()
data_epi= EpitheliumLoader()
trainer = EWCTrainer(2)


y_pred = list();
y_orig = list();

    

print("----------------------Now trianing with ewc and getting metrics ---------------------------------------->")
trainer = EWCTrainer(2)


for i in range(0,EPOCHS):
    for x, y in data_epi.load_train(0,batch_size=100):
        trainer.train_on_task("Task1 -AD HOC TASK",x,y)
        sample_data = x.clone(),y.clone()

        

trainer.train_on_task_consolidate("CONSLIDATE WEIGHTS",sample_data[0],sample_data[1])

for i in range(0,EPOCHS):
    for x, y in data_nuclei.load_train(0,batch_size=100):
        print("on main  funcition")
        print(x.shape)
        print(y.shape)
        sample_data = x.clone(),y.clone()
        trainer.train_on_task("TASK-MAIN EWC TESTING TASK",x.clone(),y.clone())
 
        

print("recording prediction of EWC Testing task")

for x,y in data_nuclei.load_test(0,batch_size=100):
    y_pred.extend(trainer.predict(x.clone()))
    y_orig.extend(y)

    

s = Score();
print("<---------------------------SCORE METRICS ----------------------------------------------------->")
print(s.build_table(y_orig,y_pred));
s.plot_roc(y_orig,y_pred)
s.plot_loss(y_orig,y_pred)

print("--------------------------SCORE METRIC END ----------------")


