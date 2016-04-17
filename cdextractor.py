#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.04.17
Description: recode-verison of cdataextractor.py to support time/structure-context
'''


import os
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import json
import cPickle
import commands

import blogger
from utils import progbar

def init_folders(dnames):
	for dname in dnames:
		if not os.path.isdir(dname):
			os.mkdir(dname)

def extract(dname_dataset, idx):
	idname = 'data/blogs/mtr/'
	dir_dataset = 'data/blogs/%s/'%(dname_dataset)
	odname = dir_dataset + 'tea/'

	init_folders([odname, dir_dataset, odname])

	ifname = idname + '%d.txt'%(idx)
	ofname = odname + '%d.pkl'%(idx)
	
	n_lines = int(commands.getoutput('grep -cF "" %s'%(ifname)))
	if n_lines == 0:
		return



	pbar = progbar.start(n_lines)
	l = 0

	datalist = []

	ofobj = open(ofname, 'w')
	with open(ifname, 'r') as ifobj:
		for line in ifobj:
			blog = json.loads(line)
			n_text = len(blog['text'])

			for i in range(1, n_text):
				res = blogger.extract(blog['text'][i])
				if res is not None:
					datum = {}
					datum['text'] = res[0]
					datum['emo'] = res[1]
					
					above_s = []			
					re_id = i
					while blog['re'][re_id] is not None:
						re_id = blog['re'][re_id]
						above_s.append(blog['text'][re_id])
					datum['above_s'] = above_s

					above_t = []
					re_id = i - 1
					while re_id >= 0:
						above_t.append(blog['text'][re_id])
						re_id -= 1
					datum['above_t'] = above_t

					ofobj.write(json.dumps(datum) + '\n')

			l += 1
			pbar.update(l)
		pbar.finish()

	ofobj.close()


def get_emotf(dname_dataset):
	emos = []
	tf = {}

	for i in range(3):
		fname = 'data/blogs/%s/tea/%d.txt'%(dname_dataset, i)
		with open(fname, 'r') as fobj:
			for line in fobj:
				blog = json.loads(line)
				emo = blog['emo']
				if tf.has_key(emo):
					tf[emo] += 1
				else:
					tf[emo] = 1

	cPickle.dump(tf, open('data/blogs/%s/emotf.pkl'%(dname_dataset), 'w'))	
	
	top_emo = sorted(tf.items(), key = lambda k:-k[1])
	emos = [emo for emo, tf in top_emo[:500]]
	open('data/blogs/%s/eid.txt'%(dname_dataset), 'w').write(u'\n'.join(emos))

def split(dname_dataset, eids):
	dir_dataset = 'data/blogs/%s/'%(dname_dataset)

	all_emos = open(dir_dataset + 'eid.txt', 'r').read().decode('utf8').split('\n')

	eidmap = {}
	for eid in eids:
		eidmap[all_emos[eid]] = eid

	odname = dir_dataset + 'eid_data/'
	init_folders([odname, ])

	datalist = {}
	for eid in eids:
		datalist[eid] = []

	for i in range(3):
		fname = dir_dataset + 'tea/%d.txt'%(i)
		with open(fname, 'r') as fobj:
			for line in fobj:
				blog = json.loads(line)
				if eidmap.has_key(blog['emo']):
					datalist[eidmap[blog['emo']]].append(line)

	for eid, lines in datalist.items():
		fname = odname + '%d.txt'%(eid)
		fobj = open(fname, 'w')
		fobj.write(''.join(lines))
		fobj.close()

def prepare_dataset(dname_dataset, n_emo, n_samples):
	dir_dataset = 'data/blogs/%s/'%(dname_dataset)

	idname = dir_dataset + 'eid_data/'
	odname = dir_dataset + 'raw'

	init_folders([odname, ])

	for eid in range(n_emo):
		ifname = idname + '%d.txt'%(eid)
		ofname = odname + '%d.txt'%(eid)
		
		print >> sys.stderr, 'preparing %s (-> %s)'%(ifname, ofname)

		blogs = []
		with open(ifname, 'r') as ifobj:
			for line in ifobj:
				blogs.append(json.loads(line))
		
		blogs = sorted(blogs, key = lambda k: -len(k['above']))

		ofobj = open(ofname, 'w')
		for blog in blogs[:n_samples]:
			ofobj.write(json.dumps(blog) + '\n')
		ofobj.close()

