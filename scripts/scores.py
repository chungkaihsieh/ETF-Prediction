# -*- coding: utf-8 -*-
import pandas as pd
import datetime
import numpy as np


def data_helper(df,date_interval):

	data= df[(df['date'][0:]>=test_interval[0]-3)& (df['date'][0:]<=test_interval[1])]
	id_unique = data['id'].unique()
	#print(id_unique)

	ud = []
	for _id in id_unique:
		ud.append(np.diff(data[data['id']==_id]['close']))
	ud = np.sign(ud).flatten()
	#print(ud)

	close = np.array((df[(df['date'][0:]>=test_interval[0])& (df['date'][0:]<=test_interval[1])]['close']).astype(dtype='float32'))
	#print(close)

	ground_truth = np.zeros( (18,10), dtype = np.float32 )

	ud_index = 0
	close_index = 0
	for i in range(np.size(ground_truth,0)):
		for j in range(np.size(ground_truth,1)):
			if j%2==0:
				#print(ud[ud_index])
				ground_truth[i][j] = ud[ud_index]
				ud_index += 1
			else:
				#print(close[close_index])
				ground_truth[i][j] = close[close_index]
				close_index += 1
	return ground_truth


def evaluation(submission,ground_truth):

	score = 0
	weights = np.array([0.1, 0.15, 0.2, 0.25, 0.3]) 
	for i in range(np.size(ground_truth,0)):
		for j in range(np.size(ground_truth,1)):
			if j%2==0: # 預測正確得0.5				
				score += ((ground_truth[i][j]-submission[i][j])==0)*weights[j//2]*0.5
			else: # (實際價格 – 絕對值(預測價格 – 實際價格)) /實際價格)*0.5)
				score += ((ground_truth[i][j] - abs(submission[i][j]-ground_truth[i][j]))/ground_truth[i][j])*weights[j//2]*0.5
	return score



if __name__ == "__main__":
	
	# 比對日期區間
	test_interval = [20180528, 20180601] 

	# submission data
	submission = pd.read_csv('for_submission/20180528_20180601/submission_without_voting.csv', skipinitialspace=True, na_values=0)
	submission = np.round(np.array((submission.drop(['ETFid'], axis=1)), dtype = np.float32 ),2)
	#submission = np.array((submission.drop(['ETFid'], axis=1)), dtype = np.float32 )
	print(submission)
	print('\n')

	# ground truth
	df = pd.read_csv('../data/origin_data/tetfp_0601.csv', skipinitialspace=True, na_values=0)
	ground_truth = data_helper(df,test_interval)
	print(ground_truth)

	score = evaluation(submission,ground_truth)

	# 14.81172
	print(score)