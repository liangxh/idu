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

from optparser import OptionParser


def classify_GMM(train, test, covariance_type = 'diag'):
	from sklearn.mixture import GMM

	x, y = train
	ydim = np.unique(test).shape[0]
	
	classifier = GMM(n_components = ydim, covariance_type = 'diag', init_params = 'wc')
	classifier.means_ = np.array([x[y == i].mean(axis = 0) for i in range(ydim)])
	classifier.fit(x)

	x, y = test
	proba = classifier.predict_proba(x)
	return proba	

def main():
	optparser = OptionParser()
	optparser.add_option('-i', '--input', action = 'store', type = 'str', dest = 'key_input')
	optparser.add_option('-m', '--models', action = 'store', type = 'str', dest = 'keys_model')

	opts, args = optparser.parse_args()

	ifname = 'data/dataset/xvec/%s.pkl'%(opts.key_input)
	train, test = cPickle.load(open(ifname, 'r'))
	
	train = np.asarray(train)
	test = np.asarray(test)
	test_y = test[1]

	keys_model = opts.keys_model

	for key_model in keys_model:
		print 'testing model %s...'%(key_model)

		proba = None

		if key_model.startswith('gmm'):
			params = key_model.split('-')
			covariance_type = params[1] if len(params) > 1 else 'diag'	

			proba = classify_GMM(train, test, covariance_type)

		prefix = '%s_%s'%(key_input, key_model)

		fname_test = 'data/dataset/test/_test.pkl'%(prefix)
		cPickle.dump((test_y, proba), open(fname_test, 'w'))

		validatica.report(test_y, proba, 'data/dataset/test/%s'%(prefix))

if __name__ == '__main__':
	main()
