# -*- coding: utf-8 -*-
import pandas as pd
from pandas.tseries.offsets import *
import numpy as np
import itertools
import time


from collections import Counter
import time 


#sklearn
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn import ensemble
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import NuSVC

#macos matplot config
import matplotlib
matplotlib.use('TkAgg')   
import matplotlib.pyplot as plt  


#import data processing methodology
from prediction import *


if __name__ == "__main__":
	t_start = time.time()
	stocks_df = read_data('../data/tetfp_indicators.csv')
	dji_df = read_data('../data/dji_indicators.csv')
	print('[read_data] costs:' + str(time.time() - t_start) + 'secs')
	t_start = time.time()
	stocks_df_normalize = normalize(stocks_df)
	dji_df_normalize = normalize(dji_df)
	print('[normalize] costs:' + str(time.time() - t_start) + 'secs')


	# 以time_frame天為一區間進行股價預測(i.e 觀察10天股價, 預測第11天)
	# 觀察1~10天(1 ~ time_feame-day_offset) 預測 5天(day_offset)後 (i.e 第16天) 的股價

	#train_interval = [train_first_day,train_last_day] 
	#(cautions): train_first_day - time_frame >= data first day

	train_interval = [['20180101','20180504'],['20180101','20180511'],['20180101','20180518'],['20180101','20180525']]
	test_interval = [['20180507','20180511'],['20180514','20180518'],['20180521','20180525'],['20180528','20180531']]

	x_train, y_train, x_test, y_test = [None]*4, [None]*4, [None]*4, [None]*4

	# ------------------------#
	#   ud value prediction   #
	# ------------------------#
	print(' **********\n predict ud \n **********')
	t_start = time.time()
	time_frame = 5
	first_time = True

	for i in range(len(x_train)):
		x_train_ud, y_train_ud, x_test_ud, y_test_ud = data_helper_ud(stocks_df_normalize, dji_df_normalize, time_frame, train_interval[i], test_interval[i])
		x_train[i], y_train[i], x_test[i], y_test[i] = x_train_ud, y_train_ud, x_test_ud, y_test_ud


	print('[data helper -ud ] costs:' + str(time.time() - t_start) + 'secs')

	t_start = time.time()


	highest_acc = 0
	acc = 0
	for i in range(1,100):
		for l in range(len(x_train)):
			nusvc = NuSVC(nu=float(i)/100.0)
			multi_target_nusvc = MultiOutputClassifier(nusvc, n_jobs=-1)
			multi_target_nusvc.fit(x_train[l], y_train[l])
			pred_ud = multi_target_nusvc.predict(x_test[l])


			class_names = ['down','balance','up']
			# Compute confusion matrix
			cnf_matrix = confusion_matrix(y_test[l].flatten(), pred_ud.flatten())
			np.set_printoptions(precision=2)
			acc += np.trace(cnf_matrix,dtype='float32')/ np.sum(np.sum(cnf_matrix,dtype='float32'))

		acc/=len(x_train)
		if acc >= highest_acc:
			highest_acc = acc
			print('acc->'+str(acc))
			print('nu->'+str(float(i)/100.0))

			# Plot non-normalized confusion matrix
			plt.figure()
			plot_confusion_matrix(cnf_matrix, classes=class_names,
			                      title='Confusion matrix, without normalization')

		acc = 0