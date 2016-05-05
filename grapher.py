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

import re
import cPickle
import validatica

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt


def export(label_prec, title, ofname):
	plt.figure()

	ydim = len(label_prec.values()[0])

	ax = plt.subplot(1, 1, 1)
	plt.title(title)
	plt.xlabel('N')
	plt.ylabel('Precision@N')
	plt.axis([1, ydim, 0., 1.])

	for label, prec in label_prec.items():
		ax.plot(range(1, ydim + 1), prec, label = label)

	handles, labels = ax.get_legend_handles_labels()
	ax.legend(handles[::-1], labels[::-1])

	plt.savefig(ofname)

def export_by_config():
	fname_config = sys.argv[1]

	lines = open(fname_config, 'r').readlines()

	ofname = lines[0].strip()
	title = lines[1].strip()

	lprec = {}
	for l in lines[2:]:
		params = l.strip().split(' ')
		if len(params) == 2:
			lprec[params[0]] = cPickle.load(open(params[1], 'r'))
		else:
			break
	
	export(lprec, title, ofname)

def main():
	rdname = 'data/blogs/dataset/'
	idname = rdname + 'test/'
	odname = rdname + 'report/'

	if not os.path.isdir(odname):
		os.mkdir(odname)

	title = 'exp2_new'
	ofname = odname + 'exp2_new.png'

	lprec = {
		'text':cPickle.load(open(idname + 'none2532_prec.pkl', 'r')),
		'text-above_s_nb':cPickle.load(open(idname + 'snb2532_prec.pkl', 'r')),
		'text-above_s_emo':cPickle.load(open(idname + 'semo2532_prec.pkl', 'r')),
		'text-above_s_nb_emo':cPickle.load(open(idname + 'snbemean2532_prec.pkl', 'r')),
		'text-above_t_nb':cPickle.load(open(idname + 'tnb2532_prec.pkl', 'r')),
		'text-above_t_emo':cPickle.load(open(idname + 'temo2532_prec.pkl', 'r')),
		'text-above_t_nb_emo':cPickle.load(open(idname + 'tnbemean2532_prec.pkl', 'r')),
	}

	export(lprec, title, ofname)

if __name__ == '__main__':
	#main()
	export_by_config()


