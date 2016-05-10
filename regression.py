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

from sklearn import svm
from sklearn.linear_model import *

def load_data(key_regdata):
	fname = 'data/dataset/regdata/%s.pkl'%(key_regdata)
	return cPickle.load(open(fname, 'r'))

def main():
	key_regdata = sys.argv[1]
	keys_model = sys.argv[2]
	flag_split = len(sys.argv) > 3 and sys.argv[3] == 's'
	
	if flag_split:
		print 'split_model'

	print '=========== %s ============='%(key_regdata)
	print 'model\tr_mean\tr_std\tp_mean\tp_std\ta_mean\ta_std'

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
		if model_class.has_key(key_model):
			model = model_class[key_model]()
		elif key_model.startswith('svm'):
			params = key_model.split('-')
			if len(params) == 1:
				model = svm.SVC()
			else:
				model = svm.SVC(kernel = params[1])

		return model

	for key_model in keys_model.split(','):
		train, test = load_data(key_regdata)
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

		dif = pred_y - test_y
	
		mean_dif = np.mean(dif, axis = 0)

		d0_std = np.std(dif[:, 1])
		d1_std = np.std(dif[:, 0])

		r = np.sqrt(np.sum(dif ** 2, axis = 1))
		r_mean = np.mean(r)
		r_std = np.std(r)

		#print '# %s-%s'%(key_regdata, key_model)
		#print 'r mean: %.6f std: %.6f'%(r_mean, r_std)
		#print 'p mean: %.6f std: %.6f'%(mean_dif[0], d0_std)
		#print 'a mean: %.6f std: %.6f'%(mean_dif[1], d1_std)

		print '%s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f'%(key_model, r_mean, r_std, mean_dif[0], d0_std, mean_dif[1], d1_std)

if __name__ == '__main__':
	main()
