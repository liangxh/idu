#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.05.15
'''

import os
import sys
reload(sys)
sys.setdefaultencoding('utf8')

import time
import cPickle
import numpy as np
from optparse import OptionParser
from sklearn.mixture import GMM

import validatica
from utils import progbar

odname = 'data/dataset/gmmubm/'

def build_ubm(x, key_input, n_components):
	#ifname = 'data/dataset/xvec/%s.pkl'%(key_input)
	#train, test = cPickle.load(open(ifname, 'r'))

	#x, y = train
	#x = np.asarray(x)
	ubm = GMM(n_components = n_components)	
	ubm.fit(x)

	if not os.path.isdir(odname):
		os.mkdir(odname)

	ofname = odname + '%s_%d.pkl'%(key_input, n_components)
	cPickle.dump(ubm, open(ofname, 'w'))

	return ubm

def load_ubm(key_input, n_components):
	ofname = odname + '%s_%d.pkl'%(key_input, n_components)

	if os.path.exists(ofname):
		print >> sys.stderr, 'load_ubm: [info] ubm at %s found'%(ofname)
		return cPickle.load(open(ofname, 'r'))
	else:
		return None

def classify(train, test, ubm, gamma = 1., r = 16.,  w = 1., m = 1., v = 1., n_components = 8):
	x, y = train
	ydim = np.unique(y).shape[0]

	M = n_components

	gs = []

	print >> sys.stderr, 'classify: [info] building gmm for each label ...'

	pbar = progbar.start(ydim)
	for i in range(ydim):
		xi = x[y == i]                  # matrix[T x xdim]
		T = xi.shape[0]

		weights = ubm.weights_          # matrix[M, ]
		probs = ubm.predict_proba(xi)   # matrix[T x M]

		Pr_t_i = probs * weights
		Pr_t_i = Pr_t_i / np.asmatrix(Pr_t_i.sum(axis = 1)).T    # matrix[T x M]

		n_i = np.asarray(Pr_t_i.sum(axis = 0)).flatten()      # matrix[M, ]
		Ex_i = np.asarray([(np.asarray(Pr_t_i[:, i]) * xi).sum(axis = 0) / n_i[i] for i in range(M)])
		# matrix[M x xdim]
		Ex2_i = np.asarray([(np.asarray(Pr_t_i[:, i]) * (xi ** 2)).sum(axis = 0) / n_i[i] for i in range(M)])
		# matrix[M x xdim] 

		alpha = lambda p: n_i / (n_i + r ** p)
		alpha_w = alpha(w)
		alpha_m = alpha(m)
		alpha_v = alpha(v)		

		# weights: matrix[M, ]
		new_weights = (alpha_w * n_i / T + (1. - alpha_w) * ubm.weights_) * gamma

		# means: matrix[M, xdim]
		alpha_m = np.asarray(np.asmatrix(alpha_m).T)
		new_means = alpha_m * Ex_i + (1. - alpha_m) * ubm.means_

		# covar: matrix[M, xdim]
		alpha_v = np.asarray(np.asmatrix(alpha_v).T)
		new_covars = alpha_v * Ex2_i + (1. - alpha_v) * (ubm.covars_ + ubm.means_ **2) - new_means ** 2

		g = GMM(n_components = M)
		g.means_ = new_means
		g.weights_ = new_weights
		g.covars_ = new_covars

		gs.append(g)

		pbar.update(i + 1)
	pbar.finish()

	x, y = test
	scores = [g.score(x) for g in gs]
	proba = np.column_stack(scores) # not probability really	

	return proba

def main():
	optparser = OptionParser()
	optparser.add_option('-i', '--input', action = 'store', type = 'str', dest = 'key_input')
	optparser.add_option('-d', '--db', action = 'store_true', dest = 'flag_db', default = False)
	optparser.add_option('-g', '--value_gamma', action = 'store', type = 'float', dest = 'value_gamma', default = 1.)
	optparser.add_option('-w', '--value_w', action = 'store', type = 'float', dest = 'value_w', default = 1.)
	optparser.add_option('-m', '--value_m', action = 'store', type = 'float', dest = 'value_m', default = 1.)
	optparser.add_option('-v', '--value_v', action = 'store', type = 'float', dest = 'value_v', default = 1.)
	optparser.add_option('-r', '--value_r', action = 'store', type = 'float', dest = 'value_r', default = 8.)
	optparser.add_option('-n', '--n_components', action = 'store', type = 'int', dest = 'n_components', default = 8)

	opts, args = optparser.parse_args()

	key_model = 'gmmubm'

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


	key_ubm = 'db_' if opts.flag_db else '' + opts.key_input
	ubm = load_ubm(key_ubm, opts.n_components)
	if ubm is None:
		ubm = build_ubm(train[0], opts.key_input, opts.n_components)

	test_y = test[1]
	proba = classify(train, test,
			ubm = ubm,
			gamma = opts.value_gamma, 
			r = opts.value_r,
			w = opts.value_w,
			m = opts.value_m,
			v = opts.value_v,
			n_components = opts.n_components,
		)

	prefix = '%s_%s'%(opts.key_input, key_model)
	
	fname_test = 'data/dataset/test/%s_test.pkl'%(prefix)
	cPickle.dump((test_y, proba), open(fname_test, 'w'))

	acc = validatica.report(test_y, proba, 'data/dataset/test/%s'%(prefix))
	print >> sys.stderr, 'Precision@N: ' + '   '.join(['(%d)%.4f'%(i + 1, acc[i]) for i in range(10)])

if __name__ == '__main__':
	main()
