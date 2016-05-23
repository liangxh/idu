#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.03.29
'''

import cPickle
import sys
reload(sys)
sys.setdefaultencoding('utf8')

from optparse import OptionParser
import datica
import validatica
from naivebayes import NaiveBayesClassifier

from utils import progbar
from const import N_EMO

import numpy as np

def main():
	optparser = OptionParser()

	# necessary
	optparser.add_option('-p', '--prefix', action='store', type = 'str', dest='prefix')
	optparser.add_option('-k', '--value_k', dest='value_k', type='float', action = 'store', default = 1.)
	optparser.add_option('-u', '--unigram', action='store_true', dest='unigram', default = False)
	optparser.add_option('-d', '--deduplicate', dest='flag_deduplicate', action = 'store_true', default = False)

	# debug
	optparser.add_option('-y', '--ydim', action='store', type='int', dest='ydim', default = N_EMO)
	optparser.add_option('-n', '--n_samples', action='store', dest='n_samples', default = None)

	opts, args = optparser.parse_args()

	if opts.unigram:
		dataset = datica.load_unigram(opts.ydim, opts.n_samples)
	else:
		dataset = datica.load_token(opts.ydim, opts.n_samples)

	def merge_train_valid(dataset):
		train, valid, test = dataset
		tx, ty = train
		vx, vy = valid
		tx.extend(vx)
		ty.extend(vy)
		return (tx, ty), test

	dataset = merge_train_valid(dataset)
	train, test = dataset

	classifier = NaiveBayesClassifier()
	classifier.train(train[0], train[1], opts.value_k, opts.flag_deduplicate)
	
	preds = [classifier.classify(x) for x in train[0]]
	prec = validatica.precision_at_n(train[1], preds)
	print prec

	#preds = [classifier.classify(x) for x in test[0]]

	#fname_test = 'data/dataset/test/%s_test.pkl'%(opts.prefix)
	#fname_valid = 'data/dataset/test/%s'%(opts.prefix)

	#cPickle.dump((test[1], preds), open(fname_test, 'w'))
	#validatica.report(test[1], preds, fname_valid)


if __name__ == '__main__':
	main()
