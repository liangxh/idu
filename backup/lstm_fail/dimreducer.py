#! /usr/bin/env python
# -*- coding: utf8 -*-
'''
Author: Xihao Liang
Created: 2016.03.15
'''

import sys
reload(sys)
sys.setdefaultencoding('utf8')

import theano
import numpy as np

from utils import dAtool

def unzip(code):
	tokens = []
	vecs = []
	for token, vec in code.items():
		tokens.append(token)
		vecs.append(vec)

	return tokens, vecs

def svd(code, dim):
	tokens, vecs = unzip(code)

	W = np.asmatrix(vecs)
	u, s, v = np.linalg.svd(W)
	

	new_code = {}
	for i, token in enumerate(tokens):
		new_code[token] = np.asarray(u[i, :dim])[0].tolist()

	return new_code

def dA(
	code,
	dim,

	# changed recommended
	corruption_level = 0.,
	training_epochs = 1000,

	# params for training
	batch_size = 64,
	learning_rate = 0.1,

	# save
	saveto = None,
	):

	tokens, vecs = unzip(code)

	def shared_dataset(data_x, borrow=True):
		return theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)

	n_visible = len(vecs[0])
	train_set_x = shared_dataset(vecs)

	da = dAtool.train(
		train_set_x = train_set_x, 
		n_visible = n_visible,
		n_hidden = dim,
		training_epochs = training_epochs,
		corruption_level = corruption_level,
	)

	if saveto is not None:
		da.dump(saveto)

	mat_code = da.get_hidden_values(np.asarray(vecs)).eval()

	new_code = {}
	for i, token in enumerate(tokens):
		new_code[token] = mat_code[i, :dim].tolist()

	return new_code

if __name__ == '__main__':
	test()
	
