#import numpy as np
#import pandas as pd
#import random
#from sklearn.preprocessing import MinMaxScaler

'''
mmscale scales a matrix between 0 and 1.
input: a matrix
output: a matrix with all values between 0 and 1.
'''
def mmscale(all_data, train_size):
	_,seq_len = all_data.shape
	training_data=all_data[:,:train_size]
	scaler = MinMaxScaler()
	scaler.fit(training_data.reshape(-1,1))
	scaled_1d = scaler.transform(all_data.reshape(-1,1))
	scaled_all = scaled_1d.reshape(-1, seq_len)
	return scaled_all

'''
mvmat produce a matrix with missing values
input: a full matrix dense_mat; the percent of missing values mv_rate
output: a matrix with mv_rate% missing values
'''

def mvmat(dense_mat, mv_rate):
	lenrow, lencol = dense_mat.shape[0], dense_mat.shape[1]
	binary_mat = np.ones(lenrow*lencol)
	lenmv = int(lenrow*lencol*mv_rate)
	binary_mat[:lenmv] = 0
	np.random.shuffle(binary_mat)
	result_mat = np.multiply(dense_mat, binary_mat.reshape(lenrow, lencol))
	return result_mat


def rmse(mat_1, mat_2):
    return np.sqrt(np.mean((mat_1-mat_2)**2))