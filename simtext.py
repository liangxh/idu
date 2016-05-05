#! /usr/bin/env python
#-*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.04.25
'''

import os
import sys
reload(sys)
sys.setdefaultencoding('utf8')

import cPickle
import Levenshtein
import numpy as np

import datica
from utils import progbar


def jacob(t1, t2):
	return - len(set(t1).intersection(set(t2)))

def sim_jacob():	
	tid =  int(sys.argv[1])

	phrases = open('data/eid.txt', 'r').read().decode('utf8').split('\n')

	n_emo = 90
	x = []
	y = []
	for eid in range(n_emo):
		seqs = cPickle.load(open('data/dataset/unigram/%d.pkl'%(eid), 'r'))
		x.extend(seqs)
		y.extend([eid for i in range(len(seqs))])

	func_distance = jacob

	x_target = x[tid]
	res = []
	pbar = progbar.start(len(x))
	for i, xi in enumerate(x):
		res.append((i, func_distance(xi, x_target)))
		pbar.update(i + 1)
	pbar.finish()

	res = sorted(res, key = lambda k:k[1])
	
	for i in range(10):
		xid, v = res[i]
		print '%d. (%d) [%s] %s'%(i, v, phrases[y[xid]], ''.join(x[xid]))	

def sim_ED():
	tid =  int(sys.argv[1])

	phrases = open('data/eid.txt', 'r').read().decode('utf8').split('\n')

	n_emo = 90
	x = []
	y = []
	for eid in range(n_emo):
		seqs = cPickle.load(open('data/dataset/unigram/%d.pkl'%(eid), 'r'))
		for seq in seqs:
			x.append(u''.join(seq))
		y.extend([eid for i in range(len(seqs))])

	func_distance = Levenshtein.distance
	
	x_target = x[tid]
	bias = len(x_target)
	res = []
	pbar = progbar.start(len(x))
	for i, xi in enumerate(x):
		res.append((i, func_distance(xi, x_target) - bias))
		pbar.update(i + 1)
	pbar.finish()

	res = sorted(res, key = lambda k:k[1])
	
	for i in range(10):
		xid, v = res[i]
		print '%d. (%d) [%s] %s'%(i, v, phrases[y[xid]], ''.join(x[xid]))

def sim_ED_batch():
	batch_num = int(sys.argv[1])
	batch_id = int(sys.argv[2])

	n_emo = 90

	dataset = datica.load_data('data/dataset/unigram/', n_emo)
	train, valid, test = dataset
	train_x, train_y = train
	test_x, test_y = test

	train_x = [u''.join(seq) for seq in train_x]
	test_x = [u''.join(seq) for seq in test_x]

	if len(test_x) % batch_num == 0:
		batch_size = len(test_x) / batch_num 
	else:
		batch_size = len(test_x) / batch_num + 1

	b = batch_id * batch_size
	e = b + batch_size
	if e > len(test_x):
		print 'ID %d ~ END'
		test_x = test_x[b:]
	else:
		print 'ID %d ~ %d'%(b, e - 1)
		test_x = test_x[b:e]

	n_test = len(test_x)
	n_train = len(train_x)

	records = []

	pbar = progbar.start(n_test * n_train)
	l = 0
	for i in range(n_test):
		target_x = test_x[i]
		target_y = test_y[i]
		len_x = len(target_x)
		
		record = []
		for xi, yi in zip(train_x, train_y):
			d = Levenshtein.distance(target_x, xi)
			if len_x - d < 1:
				record.append((yi, d))
			l += 1			
			pbar.update(l)

		records.append((target_y, len_x, record))
	
	pbar.finish()

	cPickle.dump(records, open('data/simrecord_%d_%d.pkl'%(batch_num, batch_id), 'w'))	 

if __name__ == '__main__':
	sim_ED_batch()
