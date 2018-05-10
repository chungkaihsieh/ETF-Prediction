
# -*- coding: utf-8 -*-
import pandas as pd
from pandas.tseries.offsets import *
import numpy as np
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
import time 


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
import keras

#macos matplot config
import matplotlib
matplotlib.use('TkAgg')   
import matplotlib.pyplot as plt  


def read_data(file_name):
	df = pd.read_csv(file_name, skipinitialspace=True)
	df.dropna(how='any',inplace=True)

	# #change columns name
	df.columns = ['id','date','name','open','high','low','close','volume']

	# #reorder columns
	cols = ['date','id','open','close','high', 'low','volume','name']
	df = df[cols]

	#Turn ['date'] type from string('20180130') to date(2018-01-30)
	df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='ignore')
	
	#Turn feature type to float32
	for c in cols:
		if c not in ['name','date']:
			df[c] = df[c].apply(lambda x: np.float32(x.replace(',','')) if isinstance(df.iloc[0][c], basestring) else np.float32(x))


	return df


def data_helper(df, time_frame, train_interval, test_interval, day_offset):
	# data dimensions: id, open(開盤價)、high(最高價)、low(最低價)、volume(成交量)、close(收盤價), 6 dims
	feature_cols = ['id','open','high','low','volume','close']
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
	x_train = train_result[:,:-(day_offset)-1] #Extract every time_frame -(day_offset + last day) as feature
	y_train = train_result[:,-1][:,-1] # Extract the last one time_frame close(收盤價) value as label



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
	x_test = test_result[:,:-(day_offset)-1] #Extract every time_frame -(day_offset + last day) as feature
	y_test = test_result[:,-1][:,-1] # Extract the last one time_frame close(收盤價) value as label



	return [x_train, y_train, x_test, y_test]


def build_model(input_length, input_dim):
    d = 0.3
    model = Sequential()
    model.add(LSTM(128, input_shape=(input_length, input_dim), return_sequences=True))
    model.add(Dropout(d))
    model.add(LSTM(128, input_shape=(input_length, input_dim), return_sequences=False))
    model.add(Dropout(d))
    model.add(Dense(8,kernel_initializer="uniform",activation='relu'))
    model.add(Dense(1,kernel_initializer="uniform",activation='linear'))
    model.compile(loss='mse',optimizer='adam', metrics=['accuracy'])
    return model




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


def denormalize(df, norm_value):
    original_value = df['close'].values.reshape(-1,1)
    norm_value = norm_value.reshape(-1,1)
    
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit_transform(original_value)
    denorm_value = min_max_scaler.inverse_transform(norm_value)
    
    return denorm_value


if __name__ == "__main__":
	t_start = time.time()
	stocks_df = read_data('../TBrain_Round2_DataSet_20180504/tsharep.csv')
	print('[read_data] costs:' + str(time.time() - t_start) + 'secs')
	t_start = time.time()
	stocks_df_normalize = normalize(stocks_df)

	print('[normalize] costs:' + str(time.time() - t_start) + 'secs')
	t_start = time.time()

	# 以time_frame天為一區間進行股價預測(i.e 觀察10天股價, 預測第11天)
	time_frame = 10
	day_offset = 5
	#train_interval = [train_first_day,train_last_day] 
	#(cautions): train_first_day - time_frame >= data first day
	# train_interval = ['20130201','20180504']
	train_interval = ['20180101','20180427']
	test_interval = ['20180430','20180504']
	x_train, y_train, x_test, y_test = data_helper(stocks_df, time_frame, train_interval, test_interval, day_offset)

	print('[data helper] costs:' + str(time.time() - t_start) + 'secs')
	t_start = time.time()

	# time_frame days、6 dims
	model = build_model( time_frame - day_offset, 6 )
	model.fit( x_train, y_train, batch_size=32, epochs=50, validation_split=0.1, shuffle=True, verbose=1)

	# Use trained model to predict
	pred = model.predict(x_test)
	# denormalize
	denorm_pred = denormalize(stocks_df, pred)
	denorm_ytest = denormalize(stocks_df, y_test)

	# print('pred' + str(denorm_pred))
	# print('ytest' + str(denorm_ytest))


	#matplotlib inline  
	plt.plot(denorm_pred,color='red', label='Prediction')
	plt.plot(denorm_ytest,color='blue', label='Answer')
	plt.legend(loc='best')
	plt.show()