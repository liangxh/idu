#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.05.11
'''

import sys
reload(sys)
sys.setdefaultencoding('utf8')

import cPickle
import numpy as np
from optparse import OptionParser

import validatica

def classify_GMM(train, test, covariance_type = 'diag', verbose = False):
	from sklearn.mixture import GMM

	x, y = train
	ydim = np.unique(y).shape[0]
	
	classifier = GMM(n_components = ydim, covariance_type = 'diag', init_params = 'wc', verbose = verbose)
	classifier.means_ = np.array([x[y == i].mean(axis = 0) for i in range(ydim)])
	classifier.fit(x)

	x, y = test
	proba = classifier.predict_proba(x)
	return proba

def classify_SVC(train, test, kernel = 'rbf', verbose = False):
	from sklearn.svm import SVC

	x, y = train
	classifier = SVC(probability = True, verbose = verbose)
	classifier.fit(x, y)
	
	x, y = test
	proba = classifier.predict_proba(x, kernel = kernel)
	return proba

def tovec(i, y):
	vec = np.zeros(y)
	vec[i] = 1.
	return vec

def classify_OMP(train, test):
	from sklearn.linear_model import OrthogonalMatchingPursuit as OMP

	x, y = train
	ydim = np.unique(y).shape[0]
	y = [tovec(yi, ydim) for yi in y]

	classifier = OMP()
	classifier.fit(x, y)
	
	x, y = test
	proba = classifier.predict(x)
	return proba

def main():
	optparser = OptionParser()
	optparser.add_option('-i', '--input', action = 'store', type = 'str', dest = 'key_input')
	optparser.add_option('-m', '--models', action = 'store', type = 'str', dest = 'keys_model')
	optparser.add_option('-v', '--verbose', action = 'store_true', dest = 'flag_verbose', default = False)

	opts, args = optparser.parse_args()

	ifname = 'data/dataset/xvec/%s.pkl'%(opts.key_input)
	train, test = cPickle.load(open(ifname, 'r'))

	x, y = train
	train = (np.asarray(x), np.asarray(y))
	
	x, y = test 
	test = (np.asarray(x), np.asarray(y))

	test_y = test[1]

	keys_model = opts.keys_model.split(',')

	for key_model in keys_model:
		print 'testing model %s...'%(key_model)

		proba = None

		if key_model.startswith('gmm'):
			params = key_model.split('-')
			covariance_type = params[1] if len(params) > 1 else 'diag'	

			proba = classify_GMM(train, test, covariance_type, opts.flag_verbose)
		elif key_model.startswith('svc'):
			params = key_model.split('-')
			kernel = params[1] if len(params) > 1 else 'rbf'
			
			proba = classify_SVC(train, test, kernel, opts.flag_verbose)
		elif key_model.startswith('omp'):
			proba = classify_OMP(train, test)

		prefix = '%s_%s'%(opts.key_input, key_model)

		fname_test = 'data/dataset/test/%s_test.pkl'%(prefix)
		cPickle.dump((test_y, proba), open(fname_test, 'w'))

		validatica.report(test_y, proba, 'data/dataset/test/%s'%(prefix))

if __name__ == '__main__':
	main()
