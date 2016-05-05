#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.05.05
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
			eid, prob = item
			if eid in y:
				hit[i] += 1
				break

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


def revalidate(fname_ysup, prefix):
	sups = cPickle.load(open(fname_ysup, 'r'))

	test_y, pred_probs = cPickle.load(open('data/dataset/test/%s_test.pkl'%(prefix), 'r'))
	
	y_sup = []
	for y, sup in zip(test_y, sups):
		sup.add(y)
		y_sup.append(sup)

	report(y_sup, pred_probs, 'data/dataset/test/%s_sup'%(prefix))

def main():
	thr_rate = float(sys.argv[1])
	ofname = sys.argv[2]

	n_batch = 90

	y_sup = []

	pbar = progbar.start(n_batch)
	
	for batch_id in range(n_batch):
		fname = 'data/simrecord_90_%d.pkl'%(batch_id)
		records = cPickle.load(open(fname, 'r'))

		
		for y, x_len, record in records:
			thr = x_len * thr_rate
			sup = set([yi for yi, d in record if d <= thr])
			
			y_sup.append(sup)
	
		pbar.update(batch_id + 1)

	pbar.finish()

	cPickle.dump(y_sup, open(ofname, 'w'))

if __name__ == '__main__':
	main()
