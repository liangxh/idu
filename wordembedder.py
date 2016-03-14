#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.03.14
'''

import sys
reload(sys)
sys.setdefaultencoding('utf8')

import numpy as np

def random(n_word, dim):
	'''
	by reference to lstmtool.init_params
	'''
	Wemb = 0.01 * np.random.rand(n_word, dim)
	
	return Wemb

def svd(bow, dim):
	W = np.asmatrix(bow)
	u, s, v = np.linalg.svd(W)
	return u[:, :dim]

def 

if __name__ == '__main__':
	
