#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Description: show statics about the distribution of users in the dataset
'''

import sys
reload(sys)
sys.setdefaultencoding('utf8')

def tohist(ls):
	hist = {}
	for l in ls:
		if hist.has_key(l):
			hist[l] += 1
		else:
			hist[l] = 1
	return hist

def hist2str(hist):
	return ', '.join(['%d (x%d)'%(k, v) for k, v in sorted(hist.items(), key = lambda k:-k[0])])

for i in range(100):
	mids = {}
	lines = open('data/dataset/muid/%d.txt'%(i), 'r').read().split('\n')
	for l in lines:
		params = l.split(' ')
		mid = params[0]
		uid = params[1]
		if mids.has_key(uid):
			mids[uid].append(mid)
		else:
			mids[uid] = [mid, ]

	n_mids = sorted([len(v) for v in mids.values()], reverse = True)
	mhist = tohist(n_mids)

	print '%d. %d: '%(i, len(n_mids)), hist2str(mhist)
	print 
