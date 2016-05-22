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

def main():
	'''
	a short cut to reform
	'''
	idir = sys.argv[1]
	odir = sys.argv[2]

	reform(idir, odir)

if __name__ == '__main__':
	main()
