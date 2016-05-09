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
	count_vote('050', 2)
	count_vote('050', 3)
	count_vote('075', 2)
	count_vote('075', 3)
	
