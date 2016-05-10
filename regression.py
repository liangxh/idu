#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.05.10
'''

import sys
reload(sys)
sys.setdefaultencoding('utf8')

from sklearn.linear_model import LinearRegression as LR

def load_data(key_regdata):
	fname = 'data/dataset/regdata/%s.pkl'
	return cPickle.load(open(fname, 'r'))

def linearregression(x, y):


def main():
	key_regdata = sys.argv[1]

	train, test = load_data(key_regdata)

if __name__ == '__main__':
	main()
