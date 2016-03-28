#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.03.28
'''

import os
import sys
reload(sys)
sys.setdefaultencoding('utf8')

import cPickle
import numpy as np
from sklearn.svm import SVC
from optparse import OptionParser

import validatica

def test(dataset):
	train, test = dataset
	train_x, train_y = train
	test_x, test_y = test

	clf = SVC(probability = True)
	clf.fit(train_x, train_y)

	preds = clf.predict_proba(test_x)
	return test_y, preds

def main():
	optparser = OptionParser
	optparser.add_option('-p', '--prefix', action='store', dest='prefix', type='str')
	opts, args = optparser.parse_args()

	fname_feat = 'data/dataset/feat/%s_feat.pkl'%(opts.prefix)
	if not os.path.exists(fname_feat):
		print >> sys.stderr, 'feat file %s not found'%(fname_feat)
		return

	dataset = cPickle.load(open(fname_feat, 'r'))
	train, valid, test = dataset
	tx, ty = train
	vx, vy = valid
	tx.extend(vx)
	ty.extend(vy)

	dataset = ((tx, ty), test)
	ys, preds = test(dataset)

	validatica.report(ys, preds, 'output/%sfeat_svm'%(opts.prefix))

if __name__ == '__main__':
	test('wu')

