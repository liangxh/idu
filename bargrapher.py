#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.04.15
Description: export several curves into one graph
'''

import os
import sys
reload(sys)
sys.setdefaultencoding('utf8')

import numpy as np
import re
import cPickle
import validatica

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt


def export(label_prec, title, ofname):

	colors = 'bgrcmyk'
	width = 0.15
	range_min = 1
	range_max = 5
	ind = np.arange(range_min, range_max + 1)

	plt.figure()
	ax = plt.subplot(1, 1, 1)
	plt.title(title)
	plt.xlabel('N')
	plt.ylabel('Precision@N')

	plt.axis([range_min, range_max + width * (len(label_prec) + 1), 0., 0.5])

	l = 0
	for i, item in enumerate(label_prec):
		label, prec = item
		ax.bar(ind + width * i, prec[:range_max], width, color = colors[i], label = label)
	
	ax.set_xticks(ind + width * len(label_prec) / 2)
	ax.set_xticklabels(ind)


	handles, labels = ax.get_legend_handles_labels()
	ax.legend(handles[::-1], labels[::-1])

	plt.savefig(ofname.replace('.png', '(bar).png'))

def export_by_config():
	fname_config = sys.argv[1]

	lines = open(fname_config, 'r').readlines()

	ofname = lines[0].strip()
	title = lines[1].strip()

	lprec = []
	for l in lines[2:]:
		params = l.strip().split(' ')
		if len(params) == 2:
			lprec.append((params[0], cPickle.load(open(params[1], 'r'))))
		else:
			break
	
	export(lprec, title, ofname)

if __name__ == '__main__':
	export_by_config()


