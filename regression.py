#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.05.10
'''

import sys
reload(sys)
sys.setdefaultencoding('utf8')

from sklearn.linear_model import LinearRegression as LR

def load_data(key_regdata):
	fname = 'data/dataset/regdata/%s.pkl'
	return cPickle.load(open(fname, 'r'))

def main():
	key_regdata = sys.argv[1]

	train, test = load_data(key_regdata)

	model = LR()
	model.fit(train[0], train[1])

	pred_y = model.predict(test[0])	
	test_y = np.asarray(test[1])

	dif = pred_y - test_y
	
	mean_dif = np.mean(dif, axis = 0)

	d0_std = np.std(dif[:, 1])
	d1_std = np.std(dif[:, 0])

	r = np.sqrt(np.sum(dif ** 2, axis = 1))
	r_mean = np.mean(r)
	r_std = np.std(r)

	print '# %s'%(key_regdata)
	print 'r mean: ', r_mean, ' std: ', r_std
	print 'p mean: ', mean_dif[0], 'std: ', d0_std
	print 'a mean: ', mean_dif[1], 'std: ', d1_std

if __name__ == '__main__':
	main()
