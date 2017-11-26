import matplotlib.pyplot as plt
from sklearn import metrics 
from tabulate import tabulate
#import numpy as np

#Scoring  class for printing diffrent scores and graphs.

class Score:

	def accuracy_score(self, y_orig_1, y_orig_2, y1, y2):

		return ["Accuracy Score", metrics.accuracy_score(y_orig_1, y1), metrics.accuracy_score(y_orig_2, y2)]


	def hamming_loss(self, y_orig_1, y_orig_2, y1, y2):
		
		return ["Hamming Loss", metrics.hamming_loss(y_orig_1, y1), metrics.hamming_loss(y_orig_2, y2)]


	def jaccordian_similarity(self, y_orig_1, y_orig_2, y1, y2):

		return ["Jacordian Similarity", metrics.jaccard_similarity_score(y_orig_1, y1), metrics.jaccard_similarity_score(y_orig_2, y2)]

	def log_loss(self, y_orig_1, y_orig_2 ,y1, y2):

		return ["Log Loss",metrics.log_loss(y_orig_1, y1), metrics.log_loss(y_orig_2, y2)]

	def f1_score(self, y_orig_1, y_orig_2, y1, y2):

		return ["F1 Score", metrics.f1_score(y_orig_1, y1), metrics.f1_score(y_orig_2, y2)]

	def percision_score(self, y_orig_1, y_orig_2, y1, y2):

		return ["Percision Score", metrics.precision_score(y_orig_1, y1), metrics.precision_score(y_orig_2, y2)]


	def recall_score(self,  y_orig_1, y_orig_2, y1, y2):

		return ["Recall Score", metrics.recall_score(y_orig_1, y1), metrics.recall_score(y_orig_2, y2)]

	def roc_auc_score(self, y_orig_1, y_orig_2, y1, y2):

		return ["ROC-AUC-Score", metrics.roc_auc_score(y_orig_1, y1), metrics.roc_auc_score(y_orig_2, y2)]
	
	def roc_curve(self, y_orig_1, y_orig_2, y1, y2):

		return [metrics.roc_curve(y_orig_1, y1), metrics.roc_curve(y_orig_2, y2)]
	
	#Function for plotting roc curve
	def plot_roc(self, y_orig_1, y_orig_2, y1, y2):

		roc_score = s.roc_curve(y_orig_1, y_orig_2, y1, y2)
		plt.clf();
		plt.cla();
		plt.plot(roc_score[0][0],roc_score[0][1])
		plt.savefig("roc_task1.jpg")
		plt.clf();
		plt.cla();
		plt.plot(roc_score[1][0],roc_score[1][1])
		plt.savefig("roc_task2.jpg")

	#Function for plotting loss 
	def plot_loss(self,loss_1,loss_2):

		index = [i for i in range(0, len(loss_1))]
		plt.clf();
		plt.cla();
		plt.plot(index, loss_1)
		plt.savefig("loss_task1.jpg")
		plt.clf();
		plt.cla();
		plt.plot(index, loss_2)
		plt.savefig("loss_task2.jpg")


	#generating tables style printing
	def build_table(self, y_orig_1, y_orig_2, y1, y2):

		table_data = list();
		table_data.append(self.accuracy_score(y_orig_1, y_orig_2, y1, y2))
		table_data.append(self.hamming_loss(y_orig_1, y_orig_2, y1, y2))
		table_data.append(self.jaccordian_similarity(y_orig_1, y_orig_2, y1, y2))
		table_data.append(self.log_loss(y_orig_1, y_orig_2, y1, y2))
		table_data.append(self.f1_score(y_orig_1, y_orig_2, y1, y2))
		table_data.append(self.recall_score(y_orig_1, y_orig_2, y1, y2))
		#table_data.append(self.roc_auc_score(y1,y2,y_orig))

		return tabulate(table_data,headers=["Accuracy Metric Name", "Task_1", "Task_2"])

"""
y_orig_1= np.random.randint(2, size=10)
y_orig_2= np.random.randint(2, size=10)
y1= np.random.randint(2, size=10)
y2= np.random.randint(2, size=10)

s = Score();
print("<---------------------------SCORE METRICS ----------------------------------------------------->")
print(s.build_table(y_orig_1, y_orig_2, y1, y2));
s.plot_roc(y_orig_1, y_orig_2, y1, y2)
s.plot_loss(y1, y2)
"""


