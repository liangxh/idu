#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.05.08
'''

import sys
reload(sys)
sys.setdefaultencoding('utf8')

import numpy as np
import cPickle

def main():
	p1 = sys.argv[1]
	p2 = sys.argv[2]
	ofname = sys.argv[3]

	ifname = 'data/dataset/mismatch/%s_%s.pkl'%(p1, p2)
	mispair = cPickle.load(open(ifname, 'r'))

	n_emo = 90
	
	fobj = open(ofname, 'w')

	phrases = open('data/eid.txt', 'r').read().decode('utf8').split('\n')[:n_emo]
	for i in range(n_emo):
		counts = enumerate(mispair[i, :].tolist())
		counts = sorted(counts, key = lambda k:-k[1])
		
		fobj.write('%d. %s: '%(i, phrases[i]))
		for eid, c in counts:
			if c == 0:
				break
			fobj.write('%s(%d), '%(phrases[eid], c))
		fobj.write('\n')

	fobj.close()
		
if __name__ == '__main__':
	main()
