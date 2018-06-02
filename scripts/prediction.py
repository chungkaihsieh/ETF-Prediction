# -*- coding: utf-8 -*-
import pandas as pd
from pandas.tseries.offsets import *
import numpy as np
import itertools
import time

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from collections import Counter
import time 

import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM
from keras.utils import np_utils
from keras.models import load_model
import keras.backend as K
import tensorflow as tf


#macos matplot config
import matplotlib
matplotlib.use('TkAgg')   
import matplotlib.pyplot as plt  


def read_data(file_name):
	df = pd.read_csv(file_name, skipinitialspace=True, na_values=0)
	df.fillna(value=0, inplace=True)


	if 'dji' not in file_name:
		#Turn ['date'] type from string('2018/01/30') to date(2018-01-30)
		df['date'] = pd.to_datetime(df['date'], format='%Y/%m/%d', errors='ignore')
		
		# add up/down of close value as feature
		df['ud']  = np.sign(df['close'].subtract(pd.concat([pd.Series(0), df['close'][0:-1]],ignore_index=True)))
	else:
		#Turn ['date'] type from string('20180130') to date(2018-01-30)
		df['date'] = pd.to_datetime(df['date'], format='%Y/%m/%d', errors='ignore')
		
		# add up/down of DJI's close value as feature
		df['ud_dji']  = np.sign(df['close_dji'].subtract(pd.concat([pd.Series(0), df['close_dji'][0:-1]],ignore_index=True)))


	return df




def data_helper_close(stocks_df, dji_df, time_frame, train_interval, test_interval, day_offset):
	#select and rearrange features
	# [Cautions] : 'id', 'ud', 'close' index can't be changed
	stocks_feature_cols = ['open','high','low','volume','roc','rocr100','rsi','fastk','slowk','sma','upper','macd','macdsignal','macdhist','id','ud','close']
	dji_feature_cols = ['open','high','low','volume','roc','rocr100','rsi','fastk','slowk','sma','upper','macd','macdsignal','macdhist','ud','close']
	for i in range(len(dji_feature_cols)):
		dji_feature_cols[i]+= '_dji'

	# extract unique id
 	id_unique = stocks_df['id'].unique()


	#--------------------------#
	# Training data preprocess #
	#--------------------------#
	train_end = pd.date_range(train_interval[0], train_interval[1], freq='B')
	train_result = []

	for _id in id_unique:
		# 要觀察的天數:time_frame 
		for i in range(len(train_end)):
			_time_frame = time_frame
			while(True):
				#filter data by date time interval (BDay : business day)
				stocks_df_partial = stocks_df[(stocks_df['date']>= train_end[i] - BDay(_time_frame) ) & (stocks_df['date']<= train_end[i]) & (stocks_df['id']== _id)]
				dji_df_partial = dji_df[(dji_df['date']>= train_end[i] - BDay(_time_frame) ) & (dji_df['date']<= train_end[i])]


				if len(stocks_df_partial) >= time_frame+5 and len(dji_df_partial) >= time_frame+5:
					concat_features = np.concatenate( ( dji_df_partial[-time_frame:].as_matrix(columns = dji_feature_cols),\
														stocks_df_partial[-time_frame:].as_matrix(columns = stocks_feature_cols)
										   			 ), axis = 1
													)

					train_result.append(concat_features)

					break
				_time_frame += 1
				


	train_result = np.array(train_result).astype(dtype='float32')
	x_train_close = train_result[:,:-(day_offset)] #Extract every time_frame -(day_offset + last day) as feature
	y_train_close = train_result[:,-1][:,-1] # Extract the last one time_frame close(收盤價) value as label
	# y_train_ud = train_result[:,-1][:,-2] # Extract the last one time_frame ud(收盤價漲跌) value as label


	#--------------------------#
	# Testing data preprocess #
	#--------------------------#
	test_end = pd.date_range(test_interval[0], test_interval[1], freq='B')
	test_result = []

	for _id in id_unique:
		# 要觀察的天數:time_frame 
		for i in range(len(test_end)):
			_time_frame = time_frame
			while(True):
				#filter data by date time interval (BDay : business day)
				stocks_df_partial = stocks_df[(stocks_df['date']>= test_end[i] - BDay(_time_frame) ) & (stocks_df['date']<= test_end[i]) & (stocks_df['id']== _id)]
				dji_df_partial = dji_df[(dji_df['date']>= test_end[i] - BDay(_time_frame) ) & (dji_df['date']<= test_end[i])]


				if len(stocks_df_partial) >= time_frame+5 and len(dji_df_partial) >= time_frame+5:
					concat_features = np.concatenate( ( dji_df_partial[-time_frame:].as_matrix(columns = dji_feature_cols),\
														stocks_df_partial[-time_frame:].as_matrix(columns = stocks_feature_cols)
										   			 ), axis = 1
													)
					test_result.append(concat_features)
					break
				_time_frame += 1


	test_result = np.array(test_result).astype(dtype='float32')
	x_test_close = test_result[:,:-(day_offset)] #Extract every time_frame -(day_offset + last day) as feature
	y_test_close = test_result[:,-1][:,-1] # Extract the last one time_frame close(收盤價) value as label
	# y_test_ud = test_result[:,-1][:,-2] # Extract the last one time_frame ud(收盤價漲跌) value as label


	# print('id_unique' + str(np.unique(x_train[:,0][:,0])))
	return [x_train_close, y_train_close, x_test_close, y_test_close]


def data_helper_ud(stocks_df, dji_df, time_frame, train_interval, test_interval):
	# data dimensions: id, open(開盤價)、high(最高價)、low(最低價)、volume(成交量)、close(收盤價), 6 dims
	# feature_cols = ['id','open','high','low','volume','ud','close']
	stocks_feature_cols = ['open', 'high', 'low', 'volume', 'sma', 'rsi', 'slowk', 'slowd', 'fastk', 'fastd', \
							'upper', 'middle', 'lower', 'macd', 'macdsignal', 'macdhist', 'roc', 'rocp', 'rocr100', \
							'adx', 'adxr', 'apo', 'cmo', 'mom', 'ppo', 'trix', 'id', 'ud', 'close']
	# stocks_feature_cols = ['open','high','low','volume','roc','rocr100','rsi','fastk','slowk','sma','upper','macd','macdsignal','macdhist','id','ud','close']
	dji_feature_cols = list(stocks_feature_cols)
	dji_feature_cols.remove('id')

	for i in range(len(dji_feature_cols)):
		dji_feature_cols[i]+= '_dji'
	# df = stocks_df
	# extract id
 	id_unique = stocks_df['id'].unique()
 	# print('id_unique' + str(id_unique))

	#--------------------------#
	# Training data preprocess #
	#--------------------------#
	train_end = pd.date_range(pd.to_datetime(train_interval[0]), pd.to_datetime(train_interval[1])-BDay(4), freq='B')
	train_result = []

	for _id in id_unique:
		# 要觀察的天數:time_frame 
		for i in range(len(train_end)):
			_time_frame = time_frame
			while(True):
				#filter data by date time interval (BDay : business day)
				stocks_df_partial = stocks_df[(stocks_df['date']>= train_end[i] - BDay(_time_frame) ) & (stocks_df['date']<= train_end[i]+BDay(n=4)) & (stocks_df['id']== _id)]
				dji_df_partial = dji_df[(dji_df['date']>= train_end[i] - BDay(_time_frame) ) & (dji_df['date']<= train_end[i]+BDay(n=4))]

				if len(stocks_df_partial) >= time_frame+5 and len(dji_df_partial) >= time_frame+5:
					concat_features = np.concatenate( ( dji_df_partial[-(time_frame+5):].as_matrix(columns = dji_feature_cols),\
														stocks_df_partial[-(time_frame+5):].as_matrix(columns = stocks_feature_cols)
										   			 ), axis = 1
													)

					train_result.append(concat_features)
					break
				_time_frame += 1
				


	train_result = np.array(train_result).astype(dtype='float32')
	x_train_ud = train_result[:,:-5,:].reshape(train_result.shape[0],-1) #Extract every time_frame -(day_offset + last day) as feature
	# y_train_close = train_result[:,-5:][-2] # Extract the last one time_frame close(收盤價) value as label
	y_train_ud = train_result[:,-5:,-2] # Extract the last one time_frame ud(收盤價漲跌) value as label



	#--------------------------#
	# Testing data preprocess #
	#--------------------------#
	test_end = pd.date_range(test_interval[0], test_interval[0], freq='B')
	# test_end = pd.to_datetime(test_interval[0])
	test_result = []


	for _id in id_unique:
		# 要觀察的天數:time_frame 
		for i in range(len(test_end)):
			_time_frame = time_frame
			while(True):
				#filter data by date time interval (BDay : business day)
				stocks_df_partial = stocks_df[ (stocks_df['date']<= test_end[i]+BDay(n=4)) & (stocks_df['date']>= (test_end[i] - BDay(_time_frame)) )  & (stocks_df['id']== _id)]
				dji_df_partial = dji_df[(dji_df['date']>= test_end[i] - BDay(_time_frame) ) & (dji_df['date']<= test_end[i]+BDay(n=4))]
				if len(stocks_df_partial) >= time_frame+5 and len(dji_df_partial) >= time_frame+5:
					concat_features = np.concatenate( ( dji_df_partial[-(time_frame+5):].as_matrix(columns = dji_feature_cols),\
														stocks_df_partial[-(time_frame+5):].as_matrix(columns = stocks_feature_cols)
										   			 ), axis = 1
													)

					test_result.append(concat_features)
					break
				_time_frame += 1


	test_result = np.array(test_result).astype(dtype='float32')
	x_test_ud = test_result[:,:-5,:].reshape(test_result.shape[0],-1) #Extract every time_frame -(day_offset + last day) as feature
	# y_test_close = test_result[:,-1][:,-1] # Extract the last one time_frame close(收盤價) value as label
	y_test_ud = test_result[:,-5:,-2] # Extract the last one time_frame ud(收盤價漲跌) value as label

	return [x_train_ud, y_train_ud, x_test_ud, y_test_ud]





def normalize(df):
	# newdf= df.copy()
	newdf = pd.DataFrame()
	newdf['date'] = df['date']
	cols = list(df.columns)

	# feature_cols = ['id','open','high','low','volume','roc','rocr100','rsi','fastk','slowk','sma','upper','macd','macdsignal','macdhist','ud','close']
	min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1), copy=True)
	for k in cols:
		if 'date' not in k and 'name' not in k:
			# print(k)
			newdf[k] = min_max_scaler.fit_transform(df[k].values.reshape(-1,1))


	return newdf


def denormalize(col, df, norm_value):
    original_value = df[col].values.reshape(-1,1)
    norm_value = norm_value.reshape(-1,1)    
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1), copy=True)
    min_max_scaler.fit_transform(original_value)
    denorm_value = min_max_scaler.inverse_transform(norm_value)
    
    return denorm_value


def write_csv(id, pred_ud, pred_close, file_name):
	pred_id = np.rint(id).astype(dtype=int).flatten()
	pred_close = pred_close.flatten()
	data = {'ETFid':pred_id, \
			'Mon_ud':pred_ud[:,0].flatten(), 'Mon_cprice':pred_close[0::5],\
			'Tue_ud':pred_ud[:,1].flatten(), 'Tue_cprice':pred_close[1::5],\
			'Wed_ud':pred_ud[:,2].flatten(), 'Wed_cprice':pred_close[2::5],\
			'Thu_ud':pred_ud[:,3].flatten(), 'Thu_cprice':pred_close[3::5],\
			'Fri_ud':pred_ud[:,4].flatten(), 'Fri_cprice':pred_close[4::5]}

	pred_df = pd.DataFrame(data=data, columns=['ETFid','Mon_ud', 'Mon_cprice',\
												'Tue_ud', 'Tue_cprice','Wed_ud',\
												'Wed_cprice','Thu_ud', 'Thu_cprice',\
												'Fri_ud', 'Fri_cprice'])


	#wrtie to csv
	pred_df.to_csv(file_name, index=False)




def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# 漲跌: 預測正確得0.5
# 價格: (實際價格 – 絕對值(預測價格 – 實際價格)) /實際價格)*0.5) , 也就預測越正確, 越接近0.5
def evaluation_function(denorm_pred_close, denorm_ytest_close, pred_ud, y_test_ud):

	denorm_pred_close = denorm_pred_close.flatten()
	denorm_ytest_close = denorm_ytest_close.flatten()
	pred_ud = pred_ud.flatten()
	y_test_ud = y_test_ud.flatten()

	# 價格
	value = ((denorm_ytest_close-abs(denorm_pred_close - denorm_ytest_close))/denorm_ytest_close)*0.5

	# 漲跌
	updown = np.array(np.where(pred_ud-y_test_ud==0,0.5,0))

	#every score weights per day 
	days_weights = np.array([0.1, 0.15, 0.2, 0.25, 0.3])
	value = value.flatten()
	updown = updown.flatten()

	for i in range(5):
		value[i::5] *= days_weights[i]
		updown[i::5] *= days_weights[i]

	print('close score->' + str(np.sum(value)))
	print('up score->' + str(np.sum(updown)))
	return np.sum([value + updown])




def build_model_close(input_length, input_dim):
    d = 0.2
    model = Sequential()
    model.add(LSTM(128, input_shape=(input_length, input_dim), return_sequences=False))
    model.add(Dropout(d))
    # model.add(LSTM(128, input_shape=(input_length, input_dim), return_sequences=False))
    # model.add(Dropout(d))
    # model.add(Dense(32,kernel_initializer="uniform",activation='relu'))
    # model.add(Dropout(d))
    model.add(Dense(16,kernel_initializer="uniform",activation='relu'))
    # model.add(Dropout(d))
    model.add(Dense(1,kernel_initializer="uniform",activation='linear'))
    model.compile(loss='mse',optimizer='adam')
    return model



def ud_acc(y_true, y_pred):
	tf.reshape(y_true,[-1])
	tf.reshape(y_pred,[-1])
	res = tf.equal(tf.sign(y_true),tf.sign(y_pred))
	myTensor = tf.where(res)
	countTrue = tf.shape(myTensor)[0]
	return tf.cast(countTrue, tf.float32)/tf.cast(tf.size(res),tf.float32)



def build_model_ud(input_dim, output_dim):
	d = 0.3
	model = Sequential()
	model.add(Dense(128, input_dim=input_dim, activation='relu'))
	model.add(Dropout(d/2, noise_shape=None, seed=None))
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(d, noise_shape=None, seed=None))
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(d*2, noise_shape=None, seed=None))
	model.add(Dense(output_dim, activation='softsign'))
	model.compile(loss='mean_squared_logarithmic_error',optimizer='adam', metrics=[ud_acc])
	return model



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


	# # #for testing
	train_interval = ['20180101','20180525']
	test_interval = ['20180528','20180601']

	#for submission
	# train_interval = ['20180101','20180525']
	# test_interval = ['20180528','20180601']



	#------------------------#
	# Close value prediction #
	#------------------------#

	# 以time_frame天為一區間進行股價預測(i.e 觀察10天股價, 預測第11天)
	# 觀察1~10天(1 ~ time_feame-day_offset) 預測 5天(day_offset)後 (i.e 第16天) 的股價
	time_frame = 10
	day_offset = 5
	#train_interval = [train_first_day,train_last_day] 
	#(cautions): train_first_day - time_frame >= data first day
	t_start = time.time()
	x_train_close, y_train_close, x_test_close, y_test_close = data_helper_close(stocks_df_normalize, dji_df_normalize, time_frame, train_interval, test_interval, day_offset)


	print('[data helper - close] costs:' + str(time.time() - t_start) + 'secs')
	t_start = time.time()


	# sequence length(觀察的天數~= time_frame - day_offset) ; feature length (7 dims)
	model_close = build_model_close( time_frame - day_offset, x_train_close.shape[-1] )
	earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=7)
	model_close.fit( x_train_close, y_train_close, batch_size=100, epochs=30, validation_split=0.1, shuffle=True, verbose=1, callbacks=[earlyStopping])

	# Use trained model to predict
	pred_close = model_close.predict(x_test_close)
	# denormalize
	denorm_pred_close = denormalize('close', stocks_df, pred_close)
	denorm_ytest_close = denormalize('close', stocks_df, y_test_close)




	# ------------------------#
	#   ud value prediction   #
	# ------------------------#
	print(' **********\n predict ud \n **********')
	t_start = time.time()
	time_frame = 5
	x_train_ud, y_train_ud, x_test_ud, y_test_ud = data_helper_ud(stocks_df_normalize, dji_df_normalize, time_frame, train_interval, test_interval)
	print('y_test_ud.shape->' + str(y_test_ud.shape))
	print('[data helper -ud ] costs:' + str(time.time() - t_start) + 'secs')
	t_start = time.time()



	# sequence length(觀察的天數~= time_frame - day_offset) ; feature length (7 dims)
	model_ud = build_model_ud( x_train_ud.shape[-1], y_train_ud.shape[-1] )
	earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
	# model_ud.fit( x_train_ud[::-1], y_train_ud[::-1], batch_size=5, epochs=100, shuffle=False, verbose=1, callbacks=[earlyStopping])
	model_ud.fit( x_train_ud, y_train_ud, batch_size=5, epochs=100,validation_split=0.1, shuffle=True, verbose=1, callbacks=[earlyStopping])

	# Use trained model to predict
	pred_ud = model_ud.predict(x_test_ud)
	pred_ud = np.sign(pred_ud)



	#------------------------#
	#		Evaluation		 #
	#------------------------#
	score =  evaluation_function(denorm_pred_close, denorm_ytest_close, pred_ud, y_test_ud)
	print('Total Score:' + str(score))




	#------------------------#
	#      write to csv      #
	#------------------------#

	denorm_id = denormalize('id', stocks_df, x_test_close[::5,0,-3])
	write_csv(denorm_id, pred_ud, denorm_pred_close, 'submission.csv')


	# # ------------------------#
	# #         matplot        #
	# # ------------------------#
	# #-- plot close value curve --#  
	# pred_days = 5
	# denorm_id = denormalize('id', stocks_df, x_test[:,0][:,0][::5])
	# denorm_id = np.rint(denorm_id).astype(dtype=int).flatten()

	# fig, axes = plt.subplots(3, 6, sharex='col', sharey='row')
	# axes = np.array(axes).flatten()

	# for i in range(len(denorm_pred_close)/5):
	# 	axes[i].plot(denorm_pred_close[(i)*pred_days:(i+1)*pred_days],'ro-', markersize=2.5, color='maroon',label='pred')
	# 	axes[i].plot(denorm_ytest_close[(i)*pred_days:(i+1)*pred_days],'ro-', markersize=2.5, color='cornflowerblue',label='label')
	# 	axes[i].set_title('ID:' + str(denorm_id[i]), fontsize=8)

	# 	#Change subplot edge color
	# 	plt.setp(axes[i].spines.values(), color='dimgray')
	# 	plt.setp([axes[i].get_xticklines(), axes[i].get_yticklines()], color='dimgray')
	# # Creating legend and title for the figure. Legend created with figlegend(), title with suptitle()
	# leg = fig.legend(('predict','label'),loc='upper right')

	# # Create Title, xlabel and ylabel
	# fig.suptitle('Close Value Prediction', fontsize=12, color='darkred')
	# fig.text(0.5, 0.02, 'Business Days', ha='center',fontsize=15, color='gray')
	# fig.text(0.04, 0.5, 'Close Value', va='center', rotation='vertical',fontsize=12, color='gray')


	# #--plot confusion matrix about stocks up & down --#
	class_names = ['down','balance','up']
	# Compute confusion matrix
	cnf_matrix = confusion_matrix(y_test_ud.flatten(), pred_ud.flatten())
	_ud_acc = np.trace(cnf_matrix,dtype='float32')/ np.sum(np.sum(cnf_matrix,dtype='float32'))
	print('ud accuracy :' + str(_ud_acc))
	np.set_printoptions(precision=2)

	# # Plot non-normalized confusion matrix
	# plt.figure()
	plot_confusion_matrix(cnf_matrix, classes=class_names,
	                      title='Confusion matrix, without normalization')

	# plt.show()

