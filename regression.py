#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.05.10
'''

import sys
reload(sys)
sys.setdefaultencoding('utf8')

import cPickle
import numpy as np

from sklearn.linear_model import *

def load_data(key_regdata):
	fname = 'data/dataset/regdata/%s.pkl'%(key_regdata)
	return cPickle.load(open(fname, 'r'))

def main():
	key_regdata = sys.argv[1]
	key_model = sys.argv[2]

	train, test = load_data(key_regdata)

	model_class = {
			'bayes':BayesRidge,
			'ridge':Ridge,
			'linear':LinearRegression,
			'elasticnet':ElasticNet,
			'lars':Lars,
			'lasso':Lasso,
			'lassolars':LassoLars,
		}[key_model]

	model = model_class()

	try:
		model.fit(train[0], train[1])
		pred_y = model.predict(test[0])	
	except ValueError:
		ys = []

		for i in [0, 1]:
			y = train[1][:, 0]
			model = model_class()
			model.fit(train[0], y)
		
			ys.append(model.predict(test[0]).reshape((y.shape[0], 1)))

		ys = np.concatenate(ys, axis = 1)

	test_y = np.asarray(test[1])

	dif = pred_y - test_y
	
	mean_dif = np.mean(dif, axis = 0)

	d0_std = np.std(dif[:, 1])
	d1_std = np.std(dif[:, 0])

	r = np.sqrt(np.sum(dif ** 2, axis = 1))
	r_mean = np.mean(r)
	r_std = np.std(r)

	print '# %s'%(key_regdata)
	print 'r mean: %.6f std: %.6f'%(r_mean, r_std)
	print 'p mean: %.6f std: %.6f'%(mean_dif[0], d0_std)
	print 'a mean: %.6f std: %.6f'%(mean_dif[1], d1_std)

if __name__ == '__main__':
	main()
