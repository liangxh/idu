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
	fot token, vec in code.items():
		tokens.append(token)
		vecs.append(vec)

	return tokens, vecs

def svd(code):
	tokens, vecs = unzip(code)

	W = np.asmatrix(vecs)
	u, s, v = np.linalg.svd(W)
	

	new_code = {}
	for i, token in enumerate(tokens):
		new_code[token] = np.asarray(u[i, :dim])[0].tolist()

	return new_code

def train(
	code,
	n_hidden,

	# changed recommended
	corruption_level = 0.,
	training_epochs = 15,

	# params for training
	batch_size = 64,
	learning_rate = 0.1,

	# save
	saveto = None,
	):

	vecs = code.values()

	def shared_dataset(data_x, borrow=True):
		return theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)

	n_visible = len(vecs[0])
	train_set_x = shared_dataset(vecs)

	da = dAtool.train(
		train_set_x = vecs, 
		n_visible = 28 * 28,
		n_hidden = 500,
		training_epochs = training_epochs,
		corruption_level = 0.3,
	)

	if saveto is not None:
		da.dump(saveto)

	new_code = {}
	for token, vec in code.items():
		new_code[token] = da.get_hidden_values(vec).eval()

	return new_code

def test():
	

if __name__ == '__main__':
	test()
	
