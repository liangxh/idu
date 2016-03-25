#! /usr/bin/env python
# -*- coding: utf8 -*-
'''
Author: Xihao Liang
Created: 2016.03.15
'''

import sys
reload(sys)
sys.setdefaultencoding('utf8')

import theano
import numpy as np

from utils import dAtool

def svd(Wemb, dim):
	W = np.asmatrix(Wemb)
	u, s, v = np.linalg.svd(W)
	
	return np.asarray(u[:, :dim]).astype(theano.config.floatX)

def pca(Wemb, dim):
	# Reference http://blog.csdn.net/rav009/article/details/13170725

	meanVals = np.mean(Wemb, axis=0)  
	meanRemoved = Wemb - meanVals #减去均值  
	stded = meanRemoved / np.std(Wemb) #用标准差归一化  
	covMat = np.cov(stded, rowvar=0) #求协方差方阵  
	eigVals, eigVects = np.linalg.eig(np.mat(covMat)) #求特征值和特征向量  
	eigValInd = np.argsort(eigVals)  #对特征值进行排序  
	eigValInd = eigValInd[:-(dim + 1):-1]   
	redEigVects = eigVects[:, eigValInd]       # 除去不需要的特征向量

	lowDDataMat = stded * redEigVects    #求新的数据矩阵
	#reconMat = (lowDDataMat * redEigVects.T) * np.std(Wemb) + meanVals  
	
	return lowDDataMat #, reconMat  

def dA(
	Wemb,
	dim,

	# changed recommended
	corruption_level = 0.3,
	training_epochs = 1000,

	# params for training
	batch_size = 64,
	learning_rate = 0.1,
	):

	def shared_dataset(data_x, borrow=True):
		return theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)

	n_visible = Wemb.shape[1]
	train_set_x = shared_dataset(Wemb)

	da = dAtool.train(
		train_set_x = train_set_x, 
		n_visible = n_visible,
		n_hidden = dim,
		training_epochs = training_epochs,
		corruption_level = corruption_level,
	)

	new_Wemb = da.get_hidden_values(np.asarray(Wemb)).eval()

	return new_Wemb

def test():
	import datica
	import wemb_cooc
	from wordembedder import WordEmbedder

	dataset = datica.load_unigram(2, 100)
	train_x = dataset[0][0]
	wembedder = WordEmbedder(*wemb_cooc.build(train_x))	
	#wembedder.dimreduce_load(dA(wembedder.dimreduce_prepare(), 5))
	#wembedder.dimreduce_fn(svd, 5)
	#wembedder.dimreduce_fn(dA, 500, training_epochs = 200)
	wembedder.dimreduce_fn(pca, 10)

	print wembedder.embed(train_x[0])
	print wembedder.index(train_x[0])

if __name__ == '__main__':
	test()
	
