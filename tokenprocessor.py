#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.02.26
Description: a script whichs tokenize all the text under data/dataset/text/EID.txt to data/dataset/token/EID.pkl
'''

import os
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import cPickle

import zhprocessor as zhpr
from const import DIR_TEXT, DIR_TOKEN, N_EMO
from utils import progbar

def prepare(eids = range(N_EMO)):
	if not os.path.exists(DIR_TOKEN):
		os.mkdir(DIR_TOKEN)

	pbar = progbar.start(len(eids) * 4000)
	c = 0
	for eid in eids:
		seqs = []
		lines = open(DIR_TEXT + '%d.txt'%(eid), 'r').read().split('\n')
		for line in lines:
			seqs.append(zhpr.tokenize(line))
			c += 1
			pbar.update(c)
		
		cPickle.dump(seqs, open(DIR_TOKEN + '%d.pkl'%(eid), 'w'))

	pbar.finish()

def load(n_emo = N_EMO):
	eseqs = []

	for eid in xrange(n_emo):
		fname = DIR_TOKEN + '%d.pkl'%(eid)
		if not os.path.exists(fname):
			print '[warning] %s not found, please token.prepare(EIDs)'%(fname)
			return None

		eseqs.append(cPickle.load(open(fname, 'r')))

	return eseqs	

if __name__ == '__main__':
	prepare()
