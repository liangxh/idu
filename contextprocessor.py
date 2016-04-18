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

def prepare_above_naivebayes(dname_dataset, idname, odname, n_emo, k = 1, ratio = 0.9):
	train_x = []
	train_y = []
	dlist = []


	dir_dataset = 'data/blogs/%s/'%(dname_dataset)

	idir = dir_dataset + '%s/'%(idname)
	odir = dir_dataset + '%s/'%(odname)

	init_folders([odir, ])

	print >> sys.stderr, 'contextprocessor: [info] loading data'
	for eid in range(n_emo):
		xlist = []

		ifname = idir + '%d.pkl'%(eid)
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

		print >> sys.stderr, '\t%s OK'%(ifname)

	print >> sys.stderr, 'contextprocessor: [info] training naive bayes classifier'
	classifier = NaiveBayesClassifier()
	classifier.train(train_x, train_y, k)
	
	print >> sys.stderr, 'contextprocessor: [info] exporting naive bayes result'
	for eid, xlist in enumerate(dlist):
		probs = []
		for tokens in xlist:
			probs.append(classifier.classify(tokens))
		
		ofname = odir + '%d.pkl'%(eid)
		print >> sys.stderr, '\t%s OK'%(ofname)
		cPickle.dump(probs, open(ofname, 'w'))

def prepare_above_emos(dname_dataset, idname, odname, n_emo):
	dir_dataset = 'data/blogs/%s/'%(dname_dataset)

	# get emos
	all_emos = open(dir_dataset + 'eid.txt', 'r').read().decode('utf8').split('\n')

	eidmap = {}
	for eid in range(n_emo):
		eidmap[all_emos[eid]] = eid

	idir = dir_dataset + '%s/'%(idname)
	odir = dir_dataset + '%s/'%(odname)

	init_folders([odir, ])

	print >> sys.stderr, 'contextprocessor: [info] loading data'
	for eid in range(n_emo):
		ifname = idir + '%d.pkl'%(eid)
		ofname = odir + '%d.pkl'%(eid)

		print >> sys.stderr, '\t%s -> ...'%(ifname), 

		contextu = cPickle.load(open(ifname, 'r'))
		xlist = []

		for i, comms in enumerate(contextu):
			emos = np.zeros(n_emo)
			for ts, es in comms:
				for emo in es:
					if eidmap.has_key(emo):
						emos[eidmap[emo]] += 1

			emo_sum = np.sum(emos)
			if not emo_sum == 0.:
				emos /= emo_sum

			xlist.append(emos)

		cPickle.dump(xlist, open(ofname, 'w'))
		print >> sys.stderr, '-> %s OK!'%(ofname)

def merge(dname_dataset, idnames, odname, n_emo):
	dir_dataset = 'data/blogs/%s/'%(dname_dataset)

	idirs = [dir_dataset + '%s/'%(idname) for idname in idnames]
	odir = dir_dataset + '%s/'%(odname)

	init_folders([odir, ])

	for eid in range(n_emo):
		xlist = None
		ofname = odir + '%d.pkl'%(eid)

		print >> sys.stderr, 'contextprocessor: [info] exporting to %s ...'%(ofname), 
		
		for idir in idirs:
			ifname = idir + '%d.pkl'%(eid)
			xs = cPickle.load(open(ifname, 'r'))

			if xlist is None:
				xlist = xs
			else:
				for i, x in enumerate(xs):
					xlist[i] = np.concatenate([xlist[i], x], axis = 1)

		cPickle.dump(xlist, open(ofname, 'w'))
		print >> sys.stderr, 'OK'

if __name__ == '__main__':
	pass
