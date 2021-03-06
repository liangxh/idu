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
from optparse import OptionParser

from sklearn import svm
from sklearn.linear_model import *

def load_data(key_regdata):
	fname = 'data/dataset/regdata/%s.pkl'%(key_regdata)
	return cPickle.load(open(fname, 'r'))

def main():
	optparser = OptionParser()
	optparser.add_option('-i', '--input', action = 'store', type = 'str', dest = 'key_regdata')
	optparser.add_option('-m', '--models', action = 'store', type = 'str', dest = 'keys_model')
	optparser.add_option('-s', '--split', action = 'store_true', dest = 'flag_split', default = False)
	optparser.add_option('-v', '--verbose', action = 'store_true', dest = 'flag_verbose', default = False)
	opts, args = optparser.parse_args()

	key_regdata = opts.key_regdata
	keys_model = opts.keys_model
	flag_split = opts.flag_split
	flag_verbose = opts.flag_verbose
	
	if flag_split:
		print 'split_model'

	model_class = {
			'bayes':BayesianRidge,
			'ridge':Ridge,
			'linear':LinearRegression,
			'elastic':ElasticNet,
			'lars':Lars,
			'lasso':Lasso,
			'lassolars':LassoLars,
			}


	def get_model(key_model):
		MAX_ITER = 100000

		if model_class.has_key(key_model):
			model = model_class[key_model]()
		elif key_model.startswith('svm'):
			params = key_model.split('-')
			if len(params) == 1:
				model = svm.SVR(max_iter = MAX_ITER, verbose = True)
			else:
				model = svm.SVR(kernel = params[1], max_iter = MAX_ITER, verbose = True)

		return model

	result = ''

	train, test = load_data(key_regdata)		

	def rmse(a, b):
		return np.sqrt(np.mean((a - b) ** 2))

	for key_model in keys_model.split(','):
		if flag_verbose:
			print 'test model#%s#'%(key_model)

		model = get_model(key_model)

		try:
			if flag_split:
				raise ValueError

			model.fit(train[0], train[1])
			pred_y = model.predict(test[0])	
		except ValueError:
			ys = []
			train_y = np.asarray(train[1])

			for i in [0, 1]:
				y = train_y[:, 0]
				model = get_model(key_model)
				model.fit(train[0], y)
		
				ys.append(model.predict(test[0]).reshape((len(test[0]), 1)))

			pred_y = np.concatenate(ys, axis = 1)

		test_y = np.asarray(test[1])


		p_test = test_y[:, 0]
		a_test = test_y[:, 1]
		
		p_pred = pred_y[:, 0]
		a_pred = pred_y[:, 1]

		p_coef = np.corrcoef(p_test, p_pred)[0,1]
		a_coef = np.corrcoef(a_test, a_pred)[1,0]
		
		p_rmse = rmse(p_test, p_pred)
		a_rmse = rmse(a_test, a_pred)
		
		msg = '%s\t%.4f\t%.4f\t%.4f\t%.4f'%(key_model, p_rmse, p_coef, a_rmse, a_coef)
		result += msg + '\n'

		if flag_verbose:
			print 'model#%s# done!'%(key_model)
			print msg		
	
	print '=========== %s ============='%(key_regdata)
	print 'model\tp_rmse\tp_coef\ta_rmse\ta_coef'
	print result

if __name__ == '__main__':
	main()
