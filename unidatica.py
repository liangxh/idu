#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.02.23
Description: datica for processing uni-code dataset
'''

import os
import cPickle
from tfcoder import TfCoder
from const import DIR_TEXT, PKL_TFDATA, PKL_TFCODER, N_EMO

def prepare():
	'''
	prepare data for LSTM
	data are structured as [[codes, codes, ...], ...], which are ordered by eid
	'''

	import blogger	
	from utils import progbar

	coder = cPickle.load(open(PKL_TFCODER, 'r'))

	datalist = []

	pbar = progbar.start(N_EMO)
	for eid in range(N_EMO):
		lines = open(DIR_TEXT + '%d.txt'%(eid), 'r').read().split('\n')
		data = []
		for line in lines:
			text, emo = blogger.extract(line)
			codes = coder.code(text)
			data.append(codes)
		datalist.append(data)
		pbar.update(eid + 1)

	pbar.finish()

	cPickle.dump(datalist, open(PKL_TFDATA, 'w'))

def load(n_emo = N_EMO, datalen = None, valid_rate = 0.2, test_rate = 0.1):
	'''
	load data as [[codes, codes, ...], ...] from PKL_TFDATA
	return None if something is wrong
	'''

	if not os.path.exists(PKL_TFDATA):
		print 'unidatica.load: [Remind] %s not found, please prepare() the dataset before trying to load() it'%(PKL_TFDATA)
		return None

	try:
		datalist = cPickle.load(open(PKL_TFDATA, 'r'))

		n_samples = len(datalist[0])

		if datalen is not None and n_samples > datalen:
			n_samples = datalen

		n_valid = int(valid_rate * n_samples)
		n_test = int(test_rate * n_samples)
		n_train = n_samples - n_valid - n_test

		train_x = []
		train_y = []
		for i in xrange(n_train):
			for eid in range(n_emo):
				train_x.append(datalist[eid][i])
				train_y.append(eid)

		train = (train_x, train_y)
				
		valid_x = []
		valid_y = []
		for i in xrange(n_train, n_train + n_valid):
			for eid in range(n_emo):
				valid_x.append(datalist[eid][i])
				valid_y.append(eid)

		valid = (valid_x, valid_y)

		test_x = []
		test_y = []
		for i in xrange(n_samples - n_test, n_samples):
			for eid in range(n_emo):
				test_x.append(datalist[eid][i])
				test_y.append(eid)
		
		test = (test_x, test_y)

		return train, valid, test

	except:
		print 'unidatica.load: [Error] failed to load %s'%(PKL_TFDATA)
		return None

if __name__ == '__main__':
	prepare()

