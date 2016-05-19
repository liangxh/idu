#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.05.19
'''

import os
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import cPickle
import numpy as np

import datica
from naivebayes import NaiveBayesClassifier
from const import N_EMO
from utils import progbar

def main():
	optparser = OptionParser()
	
	optparser.add_option('-x', '--dname_x', action='store', type = 'str', dest='dname_x')
	optparser.add_option('-s', '--dname_xsup', action='store', type = 'str', dest='dname_xsup')
	optparser.add_option('-k', '--value_k', dest='value_k', type='float', action = 'store', default = 1.)
	optparser.add_option('-y', '--ydim', action='store', type='int', dest='ydim', default = N_EMO)

	opts, args = optparser.parse_args()

	print >> sys.stderr, 'nbdatica: [info] loading data for training NaiveBayes ... ',
	train, valid, test = datica.load_data(opts.dname_x, opts.ydim, valid_rate = 0.)
	print >> sys.stderr, 'OK'

	print >> sys.stderr, 'nbdatica: [info] training NaiveBayes ... ',	
	classifier = NaiveBayesClassifier()
	classifier.train(train[0], train[1], opts.value_k)
	print >> sys.stderr, 'OK'

	if not os.path.exists(opts.dname_xsup):
		os.mkdir(opts.dname_xsup)

	pbar = progbar.start(opts.ydim)
	for eid in range(opts.ydim):
		ifname = opts.dname_x + '%d.pkl'%(eid)
		seqs = cPickle.load(open(ifname, 'r'))

		ofname = opts.dname_xsup + '%d.pkl'%(eid)
		proba = [classifier.classify(seq) for seq in seqs]

		cPickle.dump(proba, open(ofname, 'w'))
		pbar.update(eid + 1)
	pbar.finish()

if __name__ == '__main__':
	main()
		
