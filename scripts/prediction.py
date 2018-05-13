
# -*- coding: utf-8 -*-
import pandas as pd
from pandas.tseries.offsets import *
import numpy as np
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
import time 

import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.utils import np_utils


#macos matplot config
import matplotlib
matplotlib.use('TkAgg')   
import matplotlib.pyplot as plt  


def read_data(file_name):
	df = pd.read_csv(file_name, skipinitialspace=True)
	df.dropna(how='any',inplace=True)

	# #change columns name
	cols = ['id','date','name','open','high','low','close','volume']
	df.columns = cols

	# #reorder columns
	# cols = ['date','id','open','close','high', 'low','volume','name']
	# df = df[cols]

	#Turn ['date'] type from string('20180130') to date(2018-01-30)
	df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='ignore')
	
	#Turn feature type to float32
	for c in cols:
		if c not in ['name','date']:
			df[c] = df[c].apply(lambda x: np.float32(x.replace(',','')) if isinstance(df.iloc[0][c], basestring) else np.float32(x))


	# add up/down of close value as feature
	df['ud']  = np.sign(df['close'].subtract(pd.concat([pd.Series(0), df['close'][0:-1]],ignore_index=True)))


	return df


def data_helper(df, time_frame, train_interval, test_interval, day_offset):
	# data dimensions: id, open(開盤價)、high(最高價)、low(最低價)、volume(成交量)、close(收盤價), 6 dims
	feature_cols = ['id','open','high','low','volume','ud','close']
	number_features = len(feature_cols)
	# extract id
 	id_unique = df['id'].unique()


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
				df_partial = df[(df['date']>= train_end[i] - BDay(_time_frame) ) & (df['date']<= train_end[i]) & (df['id']== _id)]
				if len(df_partial) >= time_frame+1:
						train_result.append(df_partial[-(time_frame+1):].as_matrix(columns = feature_cols))
						break
				_time_frame += 1
				


	train_result = np.array(train_result).astype(dtype='float32')
	x_train_close = train_result[:,:-(day_offset)-1] #Extract every time_frame -(day_offset + last day) as feature
	y_train_close = train_result[:,-1][:,-1] # Extract the last one time_frame close(收盤價) value as label
	y_train_ud = train_result[:,-1][:,-2] # Extract the last one time_frame ud(收盤價漲跌) value as label


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
				df_partial = df[(df['date']>= test_end[i] - BDay(_time_frame) ) & (df['date']<= test_end[i]) & (df['id']== _id)]
				if len(df_partial) >= time_frame+1:
					test_result.append(df_partial[-(time_frame+1):].as_matrix(columns = feature_cols))
					break
				_time_frame += 1


	test_result = np.array(test_result).astype(dtype='float32')
	x_test_close = test_result[:,:-(day_offset)-1] #Extract every time_frame -(day_offset + last day) as feature
	y_test_close = test_result[:,-1][:,-1] # Extract the last one time_frame close(收盤價) value as label
	y_test_ud = test_result[:,-1][:,-2] # Extract the last one time_frame ud(收盤價漲跌) value as label


	return [x_train_close, y_train_close, y_train_ud, x_test_close, y_test_close, y_test_ud]





def normalize(df):
	newdf= df.copy()
	min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1), copy=True)
	newdf['id'] = min_max_scaler.fit_transform(df.close.values.reshape(-1,1))
	newdf['open'] = min_max_scaler.fit_transform(df.open.values.reshape(-1,1))
	newdf['low'] = min_max_scaler.fit_transform(df.low.values.reshape(-1,1))
	newdf['high'] = min_max_scaler.fit_transform(df.high.values.reshape(-1,1))
	newdf['volume'] = min_max_scaler.fit_transform(df.volume.values.reshape(-1,1))
	newdf['close'] = min_max_scaler.fit_transform(df.close.values.reshape(-1,1))

	return newdf


def denormalize(col, df, norm_value):
    original_value = df[col].values.reshape(-1,1)
    norm_value = norm_value.reshape(-1,1)
    
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit_transform(original_value)
    denorm_value = min_max_scaler.inverse_transform(norm_value)
    
    return denorm_value


def write_csv(x_test, pred_ud, pred_close, file_name):
	print('x_test.shape:{0}, pred_ud.shape:{1}, pred_close.shape:{2}'.format(x_test.shape, pred_ud.shape, pred_close.shape))
	# print(x_test[:,0][:,0], x_test[:,0][:,0].shape)
	# print(pred_ud[0::5], pred_ud[0::5].shape)

	pred_id = x_test[:,0][:,0][::5].astype(dtype=int)
	pred_ud = pred_ud.flatten()
	pred_close = pred_close.flatten()

	data = {'ETFid':pred_id, \
			'Mon_ud':pred_ud[0::5], 'Mon_cprice':pred_close[0::5],\
			'Tue_ud':pred_ud[1::5], 'Tue_cprice':pred_close[1::5],\
			'Wed_ud':pred_ud[2::5], 'Wed_cprice':pred_close[2::5],\
			'Thu_ud':pred_ud[3::5], 'Thu_cprice':pred_close[3::5],\
			'Fri_ud':pred_ud[4::5], 'Fri_cprice':pred_close[4::5]}

	pred_df = pd.DataFrame(data=data, columns=['ETFid','Mon_ud', 'Mon_cprice','Tue_ud', 'Tue_cprice','Wed_ud', 'Wed_cprice','Thu_ud', 'Thu_cprice','Fri_ud', 'Fri_cprice'])


	#wrtie to csv
	pred_df.to_csv(file_name, index=False)



def build_model_close(input_length, input_dim):
    d = 0.2
    model = Sequential()
    model.add(LSTM(128, input_shape=(input_length, input_dim), return_sequences=True))
    model.add(Dropout(d))
    model.add(LSTM(128, input_shape=(input_length, input_dim), return_sequences=False))
    model.add(Dropout(d))
    model.add(Dense(32,kernel_initializer="uniform",activation='linear'))
    model.add(Dense(16,kernel_initializer="uniform",activation='linear'))
    model.add(Dense(1,kernel_initializer="uniform",activation='sigmoid'))
    model.compile(loss='mse',optimizer='adam', metrics=['accuracy'])
    return model



def build_model_ud(input_length, input_dim):
    d = 0.2
    model = Sequential()
    model.add(LSTM(128, input_shape=(input_length, input_dim), activation='sigmoid', inner_activation='hard_sigmoid', return_sequences=True))
    model.add(Dropout(d))
    model.add(LSTM(128, input_shape=(input_length, input_dim), activation='sigmoid', inner_activation='hard_sigmoid', return_sequences=False))
    model.add(Dropout(d))
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(d))
    model.add(Dense(16,activation='relu'))
    model.add(Dropout(d)) 
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
    return model



if __name__ == "__main__":
	t_start = time.time()
	stocks_df = read_data('../TBrain_Round2_DataSet_20180504/tetfp.csv')
	print('[read_data] costs:' + str(time.time() - t_start) + 'secs')
	t_start = time.time()
	stocks_df_normalize = normalize(stocks_df)

	print('[normalize] costs:' + str(time.time() - t_start) + 'secs')
	t_start = time.time()

	# 以time_frame天為一區間進行股價預測(i.e 觀察10天股價, 預測第11天)
	# 觀察1~5天 預測 5天(day_offset)後 (i.e 第11天) 的股價
	time_frame = 10
	day_offset = 5
	#train_interval = [train_first_day,train_last_day] 
	#(cautions): train_first_day - time_frame >= data first day
	# train_interval = ['20130201','20180504']
	train_interval = ['20180101','20180427']
	test_interval = ['20180430','20180504']
	x_train, y_train_close, y_train_ud, x_test, y_test_close, y_test_ud = data_helper(stocks_df, time_frame, train_interval, test_interval, day_offset)

	print('[data helper] costs:' + str(time.time() - t_start) + 'secs')
	t_start = time.time()

	# print(y_test_ud)

	#------------------------#
	# Close value prediction #
	#------------------------#

	# time_frame days、6 dims
	model_close = build_model_close( time_frame - day_offset, 7 )
	model_close.fit( x_train, y_train_close, batch_size=32, epochs=10, validation_split=0.1, shuffle=True, verbose=1)

	# Use trained model to predict
	pred_close = model_close.predict(x_test)
	# denormalize
	denorm_pred_close = denormalize('close', stocks_df, pred_close)
	denorm_ytest_close = denormalize('close', stocks_df, y_test_close)




	#------------------------#
	#   ud value prediction  #
	#------------------------#
	# convert integers to dummy variables (i.e. one hot encoded)
	y_train_ud[y_train_ud == -1] = 0 # turn stocks drop(-1) to value(0)
	y_train_categorical_ud = np_utils.to_categorical(y_train_ud)

	# time_frame days、6 dims
	model_ud = build_model_ud( time_frame - day_offset, 7 )
	model_ud.fit( x_train, y_train_categorical_ud, batch_size=32, epochs=10, validation_split=0.1, shuffle=True, verbose=1)

	# Use trained model to predict
	pred_categorical_ud = model_ud.predict(x_test)
	#inverse value from to_categorical
	pred_ud = np.argmax(pred_categorical_ud, axis= 1)
	y_train_ud[y_train_ud == 0] = -1 # turn value(0) to stocks drop(-1)

	# denormalize
	denorm_pred_ud = denormalize('ud', stocks_df, pred_ud)
	denorm_ytest_ud = denormalize('ud', stocks_df, y_test_ud)


	#------------------------#
	#      write to csv      #
	#------------------------#
	write_csv(x_test, denorm_pred_ud, denorm_pred_close, 'submission.csv')


	# #------------------------#
	# #         matplot        #
	# #------------------------#
	# #matplotlib inline  
	# plt.plot(denorm_pred,color='red', label='Prediction')
	# plt.plot(denorm_ytest,color='blue', label='Answer')
	# plt.legend(loc='best')
	# plt.show()

	# #matplotlib inline  
	# plt.plot(denorm_pred_ud,color='red', label='Prediction')
	# plt.plot(denorm_ytest_ud,color='blue', label='Answer')
	# plt.legend(loc='best')
	# plt.show()