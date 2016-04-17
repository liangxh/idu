#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.04.13
'''

import os
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import json
import cPickle
import numpy as np

import blogger
from utils import progbar

def init_folders(dnames):
	for dname in dnames:
		if not os.path.isdir(dname):
			os.mkdir(dname)

def unigramize_dataset(dname_dataset, n_emo):
	'''
	prepare (unigramize and decompose above-follow) raw data from idname
	'''

	import zhtokenizer

	idname = 'data/blogs/%s/raw/'%(dname_dataset)
	odname_textu = 'data/blogs/%s/text_unigram/'%(dname_dataset)
	odname_aboveu = 'data/blogs/%s/above_unigram/'%(dname_dataset)
	odname_followu = 'data/blogs/%s/follow_unigram/'%(dname_dataset)

	init_folders([odname_textu, odname_aboveu, odname_followu])

	for eid in range(n_emo):
		ifname = idname + '%d.txt'%(eid)

		ofname_textu = odname_textu + '%d.pkl'%(eid)
		ofname_aboveu = odname_aboveu + '%d.pkl'%(eid)
		ofname_followu = odname_followu + '%d.pkl'%(eid)

		all_textu = []
		all_aboveu = []
		all_followu = []

		with open(ifname, 'r') as ifobj:
			for line in ifobj:
				blog = json.loads(line)

				all_textu.append(zhtokenizer.unigramize(blog['text']))

				aboveu = []
				for comm in blog['above']:
					text, emos = blogger.decompose(comm)
					aboveu.append((zhtokenizer.unigramize(text), emos))
				all_aboveu.append(aboveu)

				followu = []
				for comm in blog['follow']:
					text, emos = blogger.decompose(comm)
					followu.append((zhtokenizer.unigramize(text), emos))
				all_followu.append(followu)

		cPickle.dump(all_textu, open(ofname_textu, 'w'))
		cPickle.dump(all_aboveu, open(ofname_aboveu, 'w'))
		cPickle.dump(all_followu, open(ofname_followu, 'w'))


def load_data(n_emo, dirname_x, dirname_xsup = None, datalen = None, valid_rate = 0.2, test_rate = 0.1):
	'''
	load the dataset of EID in [0, emo) with datalen for each under dirname
	'''

	datalist_x = []

	for eid in range(n_emo):
		datalist_x.append(cPickle.load(open(dirname_x + '%d.pkl'%(eid), 'r')))
	
	n_samples = len(datalist_x[0])

	if datalen is not None and n_samples > datalen:
		n_samples = datalen

	if dirname_xsup is None:
		default_xsup = [0. ]
		datalist_xsup = [[default_xsup for i in range(n_samples)] for eid in range(n_emo)]

	else:
		datalist_xsup = []
		for eid in range(n_emo):
			datalist_xsup.append(cPickle.load(open(dirname_xsup + '%d.pkl'%(eid), 'r')))


	n_valid = int(valid_rate * n_samples)
	n_test = int(test_rate * n_samples)
	n_train = n_samples - n_valid - n_test

	def build_dataset(idx_range):
		x = []
		xsup = []
		y = []
		for i in idx_range:
			for eid in range(n_emo):
				x.append(datalist_x[eid][i])
				xsup.append(np.asarray(datalist_xsup[eid][i]))
				y.append(eid)
		return x, y, xsup

	train = build_dataset(range(n_train))
	valid = build_dataset(range(n_train, n_train + n_valid))
	test = build_dataset(range(n_samples - n_test, n_samples))

	return train, valid, test

def main():
	dname_dataset = sys.argv[1]
	n_emo = int(sys.argv[2])

	unigramize_dataset(dname_dataset, n_emo)

if __name__ == '__main__':
	main()

