import argparse
#sys.path.append("/home/adityas/Projects/DataScience2/model/")

import logging
logging.basicConfig(level=logging.DEBUG)

from keras.datasets import mnist
import numpy

import matplotlib.pyplot as plt

from sklearn import metrics 
from tabulate import tabulate
import numpy as np;


class Score:



	def accuracy_score(self,y1,y2,y_orig):

		return ["Accuracy Score",metrics.accuracy_score(y_orig,y1),metrics.accuracy_score(y_orig,y2)]


	def hamming_loss(self,y1,y2,y_orig):
		
		return ["Hamming Loss",metrics.hamming_loss(y_orig,y1),metrics.hamming_loss(y_orig,y2)]


	def classificaiton_report(self,y1,y2,y_orig):

		return [metrics.classificaiton_report(y_orig,y1),metrics.classificaiton_report(y_orig,y2)]


	def jaccordian_similarity(self,y1,y2,y_orig):

		return ["Jacordian Similarity",metrics.jaccard_similarity_score(y_orig,y1),metrics.jaccard_similarity_score(y_orig,y2)]

	def log_loss(self,y1,y2,y_orig):
		return ["Log Loss",metrics.log_loss(y_orig,y1),metrics.log_loss(y_orig,y2)]

	def f1_score(self,y1,y2,y_orig):
		return ["F1 Score",metrics.f1_score(y_orig,y1),metrics.f1_score(y_orig,y2)]

	def percision_score(self,y1,y2,y_orig):
		return ["Percision Score",metrics.precision_score(y_orig,y1),metrics.precision_score(y_orig,y2)]


	def recall_score(self,y1,y2,y_orig):
		return ["Recall Score",metrics.recall_score(y_orig,y1),metrics.recall_score(y_orig,y2)]

	def roc_auc_score(self,y1,y2,y_orig):
		return ["ROC-AUC-Score",metrics.roc_auc_score(y_orig,y1),metrics.roc_auc_score(y_orig,y2)]
	def roc_curve(self,y1,y2,y_orig):

		return [metrics.roc_curve(y_orig,y1),metrics.roc_curve(y_orig,y2)]
	

	def plot_roc(self,y_orig,y1,y2):
		roc_score = s.roc_curve(y_orig,y1,y2)
		plt.clf();
		plt.cla();
		plt.plot(roc_score[0][0],roc_score[0][1])
		plt.savefig("roc_task1.jpg")
		plt.clf();
		plt.cla();
		plt.plot(roc_score[1][0],roc_score[1][1])
		plt.savefig("roc_task2.jpg")
	def plot_loss(self,loss_1,loss_2):
		index = [i for i in range(0,len(loss_1))]
		plt.clf();
		plt.cla();
		plt.plot(index,loss_1)
		plt.savefig("loss_task1.jpg")
		plt.clf();
		plt.cla();
		plt.plot(index,loss_2)
		plt.savefig("loss_task2.jpg")



	def build_table(self,y1,y2,y_orig):
		table_data = list();
		table_data.append(self.accuracy_score(y1,y2,y_orig))
		table_data.append(self.hamming_loss(y1,y2,y_orig))
		table_data.append(self.jaccordian_similarity(y1,y2,y_orig))
		table_data.append(self.log_loss(y1,y2,y_orig))
		table_data.append(self.f1_score(y1,y2,y_orig))
		table_data.append(self.recall_score(y1,y2,y_orig))
		#table_data.append(self.roc_auc_score(y1,y2,y_orig))

		return tabulate(table_data,headers=["Accuracy Metric Name","Task_1","Task_2"])

"""
y_orig= np.random.randint(2, size=10)
y1= np.random.randint(2, size=10)
y2= np.random.randint(2, size=10)
s = Score();
print("<---------------------------SCORE METRICS ----------------------------------------------------->")
print(s.build_table(y_orig,y1,y2));
s.plot_roc(y_orig,y1,y2)
s.plot_loss(y_orig,y1)
"""

