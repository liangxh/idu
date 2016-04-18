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

from utils import progbar
from const import N_EMO

import numpy as np

def build(train_x_y, ydim):
	probwe = {}

	x, y = train_x_y
	n_y = np.zeros(ydim)

	n_samples = len(x)

	print >> sys.stderr, 'scaning train dataset'

	pbar = progbar.start(n_samples)
	loop = 0

	for seq, yi in zip(x, y):
		n_y[yi] += 1

		for token in set(seq):
			if not probwe.has_key(token):
				probwe[token] = np.zeros(ydim)
			probwe[token][yi] += 1
	
		loop += 1
		pbar.update(loop)

	pbar.finish()

	print >> sys.stderr, 'normalization'

	pbar = progbar.start(len(probwe))
	loop = 0

	for k in probwe.keys():
		probwe[k] /= n_y

		loop += 1
		pbar.update(loop)

	pbar.finish()

	return probwe

def classify(seq, ydim, probwe):
	p = np.ones(ydim)
	for t in seq:
		if not probwe.has_key(t):
			continue
		p *= probwe[t]

	return p

def classify_batch(seqs, ydim, probwe):
	return [classify(seq, ydim, probwe) for seq in seqs]

def main():
	optparser = OptionParser()

	# necessary
	optparser.add_option('-p', '--prefix', action='store', type = 'str', dest='prefix')
	optparser.add_option('-u', '--unigram', action='store_true', dest='unigram', default = False)

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

	probwe = build(train, opts.ydim)
	preds = classify_batch(test[0], opts.ydim, probwe)
	fname_test = 'data/dataset/test/%s_test.pkl'%(opts.prefix)
	fname_valid = 'data/dataset/test/%s'%(opts.prefix)

	cPickle.dump((test[1], preds), open(fname_test, 'w'))
	validatica.report(test[1], preds, fname_valid)


if __name__ == '__main__':
	main()
