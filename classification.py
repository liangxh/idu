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

def classify_GMM(train, test, covariance_type = 'diag', n_components = 8, verbose = False):
	from sklearn.mixture import GMM

	x, y = train
	ydim = np.unique(y).shape[0]
	
	gs = [GMM(n_components = n_components, covariance_type = covariance_type, init_params = 'wc')
			for i in range(ydim)
		]
	for i, g in enumerate(gs):
		g.fit(x[y == i])

	x, y = test
	scores = [g.score(x) for g in gs]
	proba = np.column_stack(scores) # not probability really	

	return proba

def classify_SVC(train, test, kernel = 'rbf', verbose = False):
	from sklearn.svm import SVC

	x, y = train
	clf = SVC(probability = True, verbose = verbose)
	clf.fit(x, y)
	
	x, y = test
	proba = clf.predict_proba(x, kernel = kernel)
	return proba

def classify_RandomForest(train, test):
	from sklearn.ensemble import RandomForestClassifier as RFC

	x, y = train
	clf = RFC()
	clf.fit(x, y)
	
	x, y = test
	proba = clf.predict_proba(x)
	return proba

def classify_AdaBoost(train, test):
	from sklearn.ensemble import AdaBoostClassifier as ABC

	x, y = train
	clf = ABC()
	clf.fit(x, y)
	
	x, y = test
	proba = clf.predict_proba(x)
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

	clf = OMP()
	clf.fit(x, y)
	
	x, y = test
	proba = clf.predict(x)
	return proba

def main():
	optparser = OptionParser()
	optparser.add_option('-i', '--input', action = 'store', type = 'str', dest = 'key_input')
	optparser.add_option('-m', '--models', action = 'store', type = 'str', dest = 'keys_model')
	optparser.add_option('-v', '--verbose', action = 'store_true', dest = 'flag_verbose', default = False)

	opts, args = optparser.parse_args()


	if opts.key_input.startswith('bow'):
		from wemb_tf import prepare_xvec
		train, test = prepare_xvec()
	else:
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
			n_components = int(params[2]) if len(params) > 2 else 1

			proba = classify_GMM(train, test, covariance_type, n_components, opts.flag_verbose)
		elif key_model.startswith('svc'):
			params = key_model.split('-')
			kernel = params[1] if len(params) > 1 else 'rbf'
			
			proba = classify_SVC(train, test, kernel, opts.flag_verbose)
		elif key_model.startswith('omp'):
			proba = classify_OMP(train, test)
		elif key_model.startswith('rf'):
			proba = classify_RandomForest(train, test)
		elif key_model.startswith('ada'):
			proba = classify_AdaBoost(train, test)
		else:
			print >> sys.stderr, '[warning] model #%s not supported'%(key_model)
			continue

		prefix = '%s_%s'%(opts.key_input, key_model)

		fname_test = 'data/dataset/test/%s_test.pkl'%(prefix)
		cPickle.dump((test_y, proba), open(fname_test, 'w'))

		acc = validatica.report(test_y, proba, 'data/dataset/test/%s'%(prefix))
		print >> sys.stderr, '%s-%s Precision@N: '%(opts.key_input, key_model) + '   '.join(['(%d)%.4f'%(i + 1, acc[i]) for i in range(10)])

if __name__ == '__main__':
	main()
