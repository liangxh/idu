#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.04.15
Description: export several curves into one graph
'''

import re
import os
import cPickle
import validatica

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt


def export(label_prec, title, ofname):
	plt.figure()

	ydim = len(label_prec.values()[0])

	ax = plt.subplot(1, 1, 1)
	plt.xlabel('N')
	plt.ylabel('Precision@N')
	plt.axis([1, ydim, 0., 1.])

	for label, prec in label_prec.items():
		ax.plot(range(1, ydim + 1), prec, label = label)

	handles, labels = ax.get_legend_handles_labels()
	ax.legend(handles[::-1], labels[::-1])

	plt.savefig(ofname)

def main():
	rdname = 'data/blogs/dataset/'
	idname = rdname + 'test/'
	odname = rdname + 'report/'

	if not os.path.isdir(odname):
		os.mkdir(odname)

	title = 'exp1'
	ofname = odname + 'exp1.png'

	lprec = {
		'text':cPickle.load(open(idname + 'ori25_prec.pkl', 'r')),
		'text-above':cPickle.load(open(idname + 'above25_prec.pkl', 'r'))
	}

	export(lprec, title, ofname)

if __name__ == '__main__':
	main()

