#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.05.22
Description: rebuilt-version of datica, new SL applied, new API added for different purpose
'''

import os
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import cPickle

import zhtokenizer
from utils import progbar

def reform(idir, odir):
	if not os.path.isdir(odir):
		os.mkdir(odir)
	
	n_emo = 90
	pbar = progbar.start(n_emo)

	for i in range(n_emo):
		seqs = cPickle.load(open(idir + '%d.pkl'%(i), 'r'))	
		fobj = open(odir + '%d.txt'%(i), 'w')
		content = u''
		content = u'\n'.join([u' '.join(seq) for seq in seqs])
		fobj.write(content)
		fobj.close()

		pbar.update(i + 1)
	pbar.finish()

def load(ifname):
	'''
	load one file
	'''
	content = open(ifname, 'r').read()
	seqs = [line.split(u' ') for line in content.split(u'\n')]

	return seqs

def load_data(dirname, n_emo, datalen = None, valid_rate = 0.2, test_rate = 0.1):
	'''
	load the dataset of EID in [0, emo) with datalen for each under dirname
	'''

	datalist = []

	for eid in range(n_emo):
		fname = dirname + '%d.txt'%(eid)
		datalist.append(load(fname))
	
	n_samples = len(datalist[0])

	if datalen is not None and n_samples > datalen:
		n_samples = datalen

	n_valid = int(valid_rate * n_samples)
	n_test = int(test_rate * n_samples)
	n_train = n_samples - n_valid - n_test

	def build_dataset(idx_range):
		x = []
		y = []
		for i in idx_range:
			for eid in range(n_emo):
				if len(datalist[eid][i]) == 0:
					# this is a bug from zhtokenizer.tokenize, not solved now 
					continue

				x.append(datalist[eid][i])
				y.append(eid)
		return x, y

	train = build_dataset(range(n_train))

	valid = build_dataset(range(n_train, n_train + n_valid)) if n_valid > 0 else None
	test = build_dataset(range(n_samples - n_test, n_samples)) if n_test > 0 else None

	return train, valid, test

def main():
	'''
	a short cut to reform
	'''
	idir = sys.argv[1]
	odir = sys.argv[2]

	reform(idir, odir)

if __name__ == '__main__':
	main()
