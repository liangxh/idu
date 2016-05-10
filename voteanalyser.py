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
			return 0

		d = np.diff(seq)
		min_d = 0.
		w = 0
		
		for i, di in enumerate(d):
			if di <= min_d:
				w = i
		return w + 1

def vote_width(key, title, ofname):
	fname = 'data/dataset/ysup/vote_%s.pkl'%(key)
	yhists = cPickle.load(open(fname, 'r'))
	
	widths = []
	for yhist in yhists:
		f_seq = np.asarray([f for y, f in yhist])
		widths.append(get_width(f_seq))

	#whist = tohist(widths)

	plt.figure()
	plt.title(title)
	plt.xlabel('Width')
	plt.ylabel('Number of Samples')
	plt.hist(widths, bins = np.arange(0, max(widths), 1))
	plt.savefig(ofname)
	
	

def count_vote(key, thr_rank):
	fname = 'data/dataset/ysup/vote_%s.pkl'%(key)
	print fname, thr_rank
	yhists = cPickle.load(open(fname, 'r'))
	
	counts = []
	for yhist in yhists:
		if len(yhist) > thr_rank:
			yhist = yhist[:thr_rank]
	
		counts.append(sum([f for y, f in yhist]))
	
	print np.mean(counts)

def main():
	pass

if __name__ == '__main__':
	#main()
	
	'''count_vote('050', 2)
	count_vote('050', 3)
	count_vote('075', 2)
	count_vote('075', 3)'''

	vote_width('075', 'thr_ED = 0.75', 'data/dataset/ysup/votewidth_075.png')
	vote_width('050', 'thr_ED = 0.50', 'data/dataset/ysup/votewidth_050.png')
	
