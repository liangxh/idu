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

from const import N_EMO, DIR_MODEL, DIR_TEST, DIR_DATA, DIR_UNIGRAM
from naivebayes import NaiveBayesClassifier

import numpy as np

def main():
	optparser = OptionParser()

	# necessary
	optparser.add_option('-p', '--prefix', action='store', type = 'str', dest='prefix')
	optparser.add_option('-x', '--dir_x', action='store', type = 'str', dest='dir_x')
	optparser.add_option('-y', '--ydim', action='store', type='int', dest='ydim')
	optparser.add_option('-n', '--n_samples', action='store', dest='n_samples', default = None)

	optparser.add_option('-k', '--value_k', dest='value_k', type='float', action = 'store', default = 1.)

	# debug
	optparser.add_option('-n', '--n_samples', action='store', dest='n_samples', default = None)

	opts, args = optparser.parse_args()
	
	#################### Preparation of Input ##############
	print >> sys.stderr, 'lstmscript.run: [info] loading dataset ... ', 
	
	n_emo = opts.ydim
	datalen = opts.n_samples		
	dataset = datica.load_data(opts.dir_x, opts.ydim, datalen) 

	print >> sys.stderr, 'Done'

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
	classifier.train(train[0], train[1], opts.value_k)
	preds = [classifier.classify(x) for x in test[0]]

	fname_test = 'data/dataset/test/%s_test.pkl'%(opts.prefix)
	fname_valid = 'data/dataset/test/%s'%(opts.prefix)

	cPickle.dump((test[1], preds), open(fname_test, 'w'))
	validatica.report(test[1], preds, fname_valid)


if __name__ == '__main__':
	main()
