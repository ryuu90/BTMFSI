import numpy as np
import seaborn as sns
import matplotlib.pyplot as  plt
import numpy as np
import pandas as pd
import random

'''
mmscale scales a matrix between 0 and 1.
input: a matrix
output: a matrix with all values between 0 and 1.
'''
def mmscale(dense_mat):
	val_max = dense_mat.max()
	val_min = dense_mat.min()
	result_mat = ((dense_mat - val_min)/(val_max - val_min))
	return result_mat

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