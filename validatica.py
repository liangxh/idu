#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.03.08
'''

import sys
reload(sys)
sys.setdefaultencoding('utf8')

import cPickle
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from utils import progbar

def precision_at_n(ys, pred_probs):
	n_test = len(ys)
	y_dim = len(pred_probs[0])
	hit = [0 for i in range(y_dim)]

	for y, probs in zip(ys, pred_probs):
		eid_prob = sorted(enumerate(probs), key = lambda k:-k[1])

		for i, item in enumerate(eid_prob):
			eid, progs = item
			if y == eid:
				hit[i] += 1

	for i in range(1, y_dim):
		hit[i] += hit[i - 1]
	
	acc = [float(hi) / n_test for hi in hit]
	return acc

def report(ys, pred_probs, prefix):
	'''
	analyse the result of test set after model training
	'''
	acc = precision_at_n(ys, pred_probs)
	cPickle.dump(acc, open('%s_prec.pkl'%(prefix), 'w'))

	y_dim = len(pred_probs[0])

	plt.figure()
	plt.axis([1, y_dim, 0., 1.])
	plt.xlabel('Top N')
	plt.ylabel('Precision')
	plt.plot(range(1, y_dim + 1), acc)

	#rand_x = range(1, y_dim + 1)
	#rand_y = [float(xi) / y_dim for xi in rand_x]
	#plt.plot(rand_x, rand_y, '--r')

	plt.savefig('%s_precision.png'%(prefix))

def report_from_file(ifname, prefix):
	'''
	demo
	'''

	import cPickle
	test_y, pred_probs = cPickle.load(open(ifname, 'r'))
	
	report(test_y, pred_probs, prefix)

if __name__ == '__main__':
	report_from_file
