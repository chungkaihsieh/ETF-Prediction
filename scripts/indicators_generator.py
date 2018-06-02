# -*- coding: utf-8 -*-
import talib
import numpy as np 
import pandas as pd 
from talib.abstract import *
from talib import MA_Type
import sys


#----------------------------------------------------------------------------
# Usage : python indicators_generator.py input.csv output.csv mark(i.e _dji)
#--------------------------------------------------------------------------



input_file = sys.argv[1]
output_file = sys.argv[2]
mark = sys.argv[3]

df = pd.read_csv(input_file, encoding='Big5')

data_start_time = '20170101'
#set date as index
if mark == '':
	df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='ignore')
	df.set_index(pd.DatetimeIndex(df['date']), inplace=True)
	df = df[df['date'] >= pd.to_datetime(data_start_time)]
else:
	df['date'] = pd.to_datetime(df['date'], format='%Y/%m/%d', errors='ignore')
	df.set_index(pd.DatetimeIndex(df['date']), inplace=True)	
	df = df[df['date'] >= pd.to_datetime(data_start_time)]

# # note that all ndarrays must be the same length!
# inputs = {
#     'open': np.random.random(100),
#     'high': np.random.random(100),
#     'low': np.random.random(100),
#     'close': np.random.random(100),
#     'volume': np.random.random(100)
# }


timeperiod = 5


##########################
# [necessary] indicators #
##########################
#moving average
df['sma'] = talib.SMA(df['close'])
df['rsi'] = RSI(df)
df['slowk'], df['slowd'] = talib.STOCH(df['high'], df['low'], df['close'], fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
# df['stochf'] = STOCHF(df)
df['fastk'], df['fastd'] = talib.STOCHRSI(df['close'], timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
df['upper'], df['middle'], df['lower'] = talib.BBANDS(df['close'], matype=MA_Type.T3)
# df['macd'],df['macdsignal'],df['macdhist'] = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
df['macd'],df['macdsignal'],df['macdhist'] = talib.MACD(df['close'], fastperiod=6, slowperiod=13, signalperiod=5)



#########################
# [Optional] indicators #  
#########################

# df['roc'] = ROC(df, timeperiod = 10) 
df['roc'] = ROC(df, timeperiod = 5)
df['rocp'] = ROCP(df, timeperiod = 5)
df['rocr100'] = ROCR100(df, timeperiod = 5)


# df['adx'] = ADX(df, timeperiod = 14)
df['adx'] = ADX(df, timeperiod = 7)
# df['adxr' ] = talib.ADXR(df['high'],df['low'],df['close'],timeperiod = 14)
df['adxr'] = talib.ADXR(df['high'],df['low'],df['close'],timeperiod = 7)


# df['apo'] = talib.APO(df['close'], fastperiod=12, slowperiod=26, matype=0)
df['apo'] = talib.APO(df['close'], fastperiod=6, slowperiod=13, matype=0)

# df['cmo'] = talib.CMO(df['close'], timeperiod=14)
df['cmo'] = talib.CMO(df['close'], timeperiod=7)

# df['mom'] = talib.MOM(df['close'], timeperiod=10)
df['mom'] = talib.MOM(df['close'], timeperiod=5)

# df['ppo'] = talib.PPO(df['close'], fastperiod=12, slowperiod=26, matype=0)
df['ppo'] = talib.PPO(df['close'], fastperiod=6, slowperiod=13, matype=0)

# df['trix'] = talib.TRIX(df['close'], timeperiod=30)
df['trix'] = talib.TRIX(df['close'], timeperiod=5)


# df['bop'] = BOP(df)
# df['cci'] = CCI(df)

# df['dx'] = DX(df)


# df['macdext'] = MACDEXT(df)
# df['macdfix'] = MACDFIX(df)
# df['mfi'] = MFI(df)
# df['minus_di'] = MINUS_DI(df)
# df['minus_dm'] = MINUS_DM(df)
# df['plus_di'] = PLUS_DI(df)
# df['plus_dm'] = PLUS_DM(df)


# # df['aroon_down'],df['aroonup'] = talib.AROON(df['high'],df['low'], timeperiod=14)
# df['aroon_down'],df['aroonup'] = talib.AROON(df['high'],df['low'], timeperiod=7)
# # df['aroonosc'] = talib.AROONOSC(high, low, timeperiod=14)
# df['aroonosc'] = talib.AROONOSC(high, low, timeperiod=7)

# df['trix'] = TRIX(df)
# df['ultosc'] = ULTOSC(df)
# df['willr'] = WILLR(df)



# Mark column name by _dji
cols = list(df.columns)
for i in range(len(cols)):
	cols[i] += mark

df.columns = cols


df.to_csv(output_file, encoding='utf-8', na_rep=0.0)
