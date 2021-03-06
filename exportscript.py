#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.03.29
Description: export several curves into one graph
'''

import re
import os
import cPickle
import validatica

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

dirname = 'data/dataset/report/'
if not os.path.isdir(dirname):
	os.mkdir(dirname)

def load_prec(fname):
	ys, preds = cPickle.load(open(fname, 'r'))
	return validatica.precision_at_n(ys, preds)

def load_precs(fnames):
	precs = {}
	for label, fname in fnames.items():
		precs[label] = load_prec(fname)

	return precs

def export(labels, precs, title):
	plt.figure()
	ax = plt.subplot(1, 1, 1)
	plt.xlabel('N')
	plt.ylabel('Precision')
	plt.axis([1, 90, 0., 1.])

	for label in labels:
		prec = precs[label]
		ax.plot(range(1, 91), prec, label = label)

	handles, labels = ax.get_legend_handles_labels()
	ax.legend(handles[::-1], labels[::-1])

	plt.savefig(dirname + '%s.png'%(title))

if __name__ == '__main__':


	prefixs = ['nb9010', 'nbu9010', 'randu9064', 'svdu9064', 'rand9064', 'svd9064', 'wvu9064', 'pca9064', 'pcau9064']
	
	fnames = {}
	for prefix in prefixs:
		fname = 'data/dataset/test/%s_test.pkl'%(prefix) 
		prefix = re.sub('\d', '', prefix)
		fnames[prefix] = fname

	precs = load_precs(fnames)

	print 'OK'
	export(['nbu', 'randu', 'svdu', 'wvu', 'pcau'], precs, '1')
	print 'OK'
	export(['wv', 'wvu'], precs, '2')
	print 'OK'
	export(['rand', 'randu'], precs, '3')
	print 'OK'
	export(['pca', 'pcau'], precs, '4')
	print 'OK'
	export(['svd', 'svdu'], precs, '5')

