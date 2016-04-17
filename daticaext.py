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
	prepare (unigramize and decompose above) raw data from idname
	'''

	import zhtokenizer

	dir_dataset = 'data/blogs/%s/'%(dname_dataset)
	idname = dir_dataset + 'raw/'
	odname_textu = dir_dataset + 'text_unigram/'
	odname_abovesu = dir_dataset + 'above_s_unigram/'
	odname_abovetu = dir_dataset + 'above_t_unigram/'

	init_folders([odname_textu, odname_abovesu, odname_abovetu])

	for eid in range(n_emo):
		ifname = idname + '%d.txt'%(eid)

		ofname_textu = odname_textu + '%d.pkl'%(eid)
		ofname_abovesu = odname_abovesu + '%d.pkl'%(eid)
		ofname_abovetu = odname_abovetu + '%d.pkl'%(eid)

		all_textu = []
		all_abovesu = []
		all_abovetu = []

		with open(ifname, 'r') as ifobj:
			for line in ifobj:
				blog = json.loads(line)

				all_textu.append(zhtokenizer.unigramize(blog['text']))

				abovesu = []
				for comm in blog['above_s']:
					text, emos = blogger.decompose(comm)
					abovesu.append((zhtokenizer.unigramize(text), emos))
				all_abovesu.append(abovesu)

				abovetu = []
				for comm in blog['above_t']:
					text, emos = blogger.decompose(comm)
					abovetu.append((zhtokenizer.unigramize(text), emos))
				all_abovetu.append(abovetu)

		cPickle.dump(all_textu, open(ofname_textu, 'w'))
		cPickle.dump(all_abovesu, open(ofname_abovesu, 'w'))
		cPickle.dump(all_abovetu, open(ofname_abovetu, 'w'))


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

