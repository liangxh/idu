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
import matplotlib.pyplot as plt


from utils import progbar

def analyse_result(ys, pred_probs, ofname = 'output/new_precision.png'):
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

	plt.figure()
	plt.axis([1, y_dim, 0., 1.])
	plt.xlabel('Top N')
	plt.ylabel('Precision')
	plt.plot(range(1, y_dim + 1), acc)

	rand_x = range(1, y_dim + 1)
	rand_y = [float(xi) / y_dim for xi in rand_x]
	plt.plot(rand_x, rand_y, '--r') 

	plt.savefig(ofname)

def test(ifname = 'output/lstm_result.pkl', ofname = 'output/precision.png'):
	import cPickle
	test_y, pred_probs = cPickle.load(open(ifname, 'r'))
	
	analyse_result(test_y, pred_probs, ofname)

if __name__ == '__main__':
	test()
