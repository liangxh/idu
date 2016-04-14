#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.04.14
'''

import os
import sys
reload(sys)
sys.setdefaultencoding('utf8')

import numpy as np
import cPickle
from optparse import OptionParser

from utils import progbar
from naivebayes import NaiveBayesClassifier

def init_folders(dnames):
	for dname in dnames:
		if not os.path.isdir(dname):
			os.mkdir(dname)	

def prepare_above_naivebayes(dname_dataset, n_emo, k = 1, ratio = 0.9):
	train_x = []
	train_y = []
	dlist = []

	idname = 'data/blogs/dataset/above_unigram/'

	odname = 'data/blogs/dataset/xsup_above_nb/'
	init_folders([odname, ])

	print >> sys.stderr, 'contextprocessor: [info] loading data'
	for eid in range(n_emo):
		xlist = []

		ifname = idname + '%d.pkl'%(eid)
		print >> sys.stderr, '\t%s OK'%(ifname)

		contextu = cPickle.load(open(ifname, 'r'))
		
		n_train = int(len(contextu) * ratio)

		for i, comms in enumerate(contextu):
			tokens = []
			for ts, emos in comms:
				tokens.extend(ts)

			xlist.append(tokens)

			if i < n_train:
				train_x.append(tokens)
				train_y.append(eid)

		dlist.append(xlist)

	print >> sys.stderr, 'contextprocessor: [info] training naive bayes classifier'
	classifier = NaiveBayesClassifier()
	classifier.train(train_x, train_y, k)
	
	print >> sys.stderr, 'contextprocessor: [info] exporting naive bayes result'
	for eid, xlist in enumerate(dlist):
		probs = []
		for tokens in xlist:
			probs.append(classifier.classify(tokens))
		
		ofname = odname + '%d.pkl'%(eid)
		print >> sys.stderr, '\t%s OK'%(ofname)
		cPickle.dump(probs, open(ofname, 'w'))


if __name__ == '__main__':
	pass
