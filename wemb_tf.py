#! /usr/env/python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.03.25
'''

import sys
reload(sys)
sys.setdefaultencoding('utf8')
import cPickle

import numpy as np
import theano

def build(seqs, N):
	tf = {}

	for seq in seqs:
		for t in seq:
			if tf.has_key(t):
				tf[t] += 1
			else:	
				tf[t] = 1.

	tf = sorted(tf.items(), key = lambda k: -k[1])
	
	nums = set('0123456789')
	Widx = {}
	idx = 0
	for t, f in tf:
		if not t in nums:
			Widx[t]	= idx	
			idx += 1
			if idx == N:
				break
	
	return Widx

def prepare():
	import datica
	train, valid, test = datica.load_data('data/dataset/unigram/', 90, valid_rate = 0.)

	x, y = train
	Widx = build(x, 2000)
	cPickle.dump(Widx, open('data/dataset/model/bow2000.pkl', 'w'))

if __name__ == '__main__':
	prepare()
