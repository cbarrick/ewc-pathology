# Entry point for EWC Trainer
import argparse
from model import  alexnet
import numpy
from ewctrainer.Score import Score

from datasets.nuclei import NucleiLoader
from datasets.epi import EpitheliumLoader
from ewctrainer.Trainer import Trainer

import logging

logger=logging.getLogger(__name__)

"""
Main function for calling the EWC Trainer

"""
if __name__ == "__main__":
	parser=argparse.ArgumentParser()
	parser.add_argument("-e", type = int, help = "epochs for the model")
	args=parser.parse_args()
	print(args)
	EPOCHS = args.e

	#code for loading training and testing datase for tasks 1 and 2

	data_nuclei = NucleiLoader()
	data_epi= EpitheliumLoader()
	trainer = Trainer(2)

	#place holder for storing information regarding to prediction of task 1 and 2
	y_pred_task_1 = list();
	y_orig_task_1 = list();
	y_pred_task_2 = list();
	y_orig_task_2 = list();



	print("----------------------Now trianing with ewc and getting metrics ---------------------------------------->")
	#launch trianer for 2 classes and 1 epoch per batch

	for i in range(0,EPOCHS):
		logger.info("EPOCHS on EPI loader training {}".format(EPOCHS))
		for x, y in data_nuclei.load_train(0, batch_size = 100):
			sample_data = x.clone(), y.clone()
			trainer.partial_fit("Task1 -AD HOC TASK" ,x ,y)



	trainer.train_on_task_consolidate("CONSLIDATE WEIGHTS", sample_data[0], sample_data[1])

	for i in range(0,EPOCHS):
		logger.info("EPOCHS on Nuclei loader training {}".format(EPOCHS))

		for x, y in data_epi.load_train(0, batch_size = 100):
			trainer.partial_fit("TASK-MAIN EWC TESTING TASK", x, y)





	print("recording prediction of EWC Testing task")

	for x,y in data_nuclei.load_test(0, batch_size = 100):
		y_pred_task_1.extend(trainer.predict(x))
		y_orig_task_1.extend(y)



	for x,y in data_epi.load_test(0, batch_size = 100):
		y_pred_task_2.extend(trainer.predict(x))
		y_orig_task_2.extend(y)



	s = Score();
	print("<---------------------------SCORE METRICS ----------------------------------------------------->")
	print(s.build_table(y_orig_task_1, y_orig_task_2, y_pred_task_1, y_pred_task_2));
	s.plot_roc(y_orig_1, y_orig_2, y_pred_task_1, y_pred_task_2)
	#s.plot_loss(y_orig,y_pred)

	print("--------------------------SCORE METRIC END ----------------")
