#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.05.09
'''

import sys
reload(sys)
sys.setdefaultencoding('utf8')

import numpy as np
import cPickle

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

def tohist(ls):
	hist = {}
	for l in ls:
		if hist.has_key(l):
			hist[l] += 1
		else:
			hist[l] = 1.
	
	return hist

def get_width(seq):
		if len(seq) < 2:
			return len(seq)

		d = np.diff(seq)
		min_d = 0.
		w = 0
		
		for i, di in enumerate(d):
			if di <= min_d:
				w = i
		return w + 1

'''
def get_cov(seq):
	if len(seq) < 2:
			return len(seq)

	c = []
	total = np.sum(seq)
	last = 0.
	for di in seq:
		last += di / total
		if c > 0.75:
			c.append(last)
'''	

def vote_width(key, title, ofname):
	fname = 'data/dataset/ysup/vote_%s.pkl'%(key)
	yhists = cPickle.load(open(fname, 'r'))
	
	widths = []
	for yhist in yhists:
		f_seq = np.asarray([f for y, f in yhist])
		widths.append(get_width(f_seq))

	plt.figure()
	plt.title(title)
	plt.xlabel('Number of Representative Emoticons')
	plt.ylabel('Number of Samples')
	plt.hist(widths, bins = np.arange(0, max(widths), 1), normed = True)
	plt.savefig(ofname)

	whist = tohist(widths)
	total = sum(whist.values())
	wc = []
	last = 0.
	for k, v in sorted(whist.items(), key = lambda k:k[0])[1:]:
		last = float(v) / total 
		print k, last
	
	
	

def count_vote(key, title):
	fname = 'data/dataset/ysup/vote_%s.pkl'%(key)
	yhists = cPickle.load(open(fname, 'r'))
	
	'''counts = [[] for i in range(1, 4)]

	for yhist in yhists:
		for i in range(1, 4):
			partial_yhist = yhist if len(yhist) <= i else yhist[:i]
			counts[i - 1].append(sum([f for y, f in partial_yhist]))
	
	for i in range(1, 4):
		plt.figure()
		plt.title(title + ' thr_rank=%d'%(i))
		plt.xlabel('Number of Similar Samples')
		plt.ylabel('Number of Test Samples')

		count = counts[i - 1]
		plt.hist(count)

		plt.savefig('data/dataset/ysup/votecount_%s_%d.png'%(key, i))'''

	counts = [sum([f for y, f in yhist]) for yhist in yhists]
	x_min, x_max = (1, 500)

	plt.figure()
	plt.title(title + ' (Mini)')
	plt.xlabel('Number of Similar Samples')
	plt.ylabel('Percentage of Test Samples (%)')
	plt.hist(counts, bins = np.arange(x_min, x_max, 10))

	to_percentage = lambda y, pos: str(round( ( y / 36000. ) * 100.0, 1))
	plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percentage))

	plt.savefig('data/dataset/ysup/votecount_%s_mini.png'%(key))

	plt.figure()
	plt.title(title)


	x_min, x_max = (min(counts), 1500)
	plt.axis((x_min, x_max, 0, 30600))

	plt.xlabel('Number of Similar Samples')
	plt.ylabel('Percentage of Test Samples (%)')
	plt.hist(counts, bins = np.arange(x_min, x_max, 20))
	
	to_percentage = lambda y, pos: str(round( ( y / 36000. ) * 100.0, 1))
	plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percentage))

	plt.savefig('data/dataset/ysup/votecount_%s.png'%(key))
	

def main():
	pass

if __name__ == '__main__':
	#main()
	
	'''count_vote('050', 2)
	count_vote('050', 3)
	count_vote('075', 2)
	count_vote('075', 3)'''
	
	count_vote('075', 'thr_ED=0.75')
	#count_vote('050', 'thr_ED=0.50')

	vote_width('075', 'thr_ED = 0.75', 'data/dataset/ysup/votewidth_075.png')
	#vote_width('050', 'thr_ED = 0.50', 'data/dataset/ysup/votewidth_050.png')
	
