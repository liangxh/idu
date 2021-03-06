#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.04.11
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


def extract(ifname, ofname):
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
					
					above = []			
					re_id = i
					while blog['re'][re_id] is not None:
						re_id = blog['re'][re_id]
						above.append(blog['text'][re_id])
					datum['above'] = above

					follow = []
					last_id = i					
					for j in range(i + 1, n_text):
						if blog['re'][j] == last_id:
							follow.append(blog['text'][j])
							last_id = j

					datum['follow'] = follow

					ofobj.write(json.dumps(datum) + '\n')
	
					#for k in range(n_text):
					#	print '%d. (-> %s) %s'%(k, blog['re'][k], blog['text'][k])
					#
					#print 'above:'
					#print '\n'.join(above)
					#print 'follow:'
					#print '\n'.join(follow)
					#print 

				#if i > 100:
				#	return

			l += 1
			pbar.update(l)
		pbar.finish()
	
	ofobj.close()

def get_emotf():
	emos = []
	tf = {}

	for i in range(3):
		fname = 'data/blogs/teaf/%d.txt'%(i)
		with open(fname, 'r') as fobj:
			for line in fobj:
				blog = json.loads(line)
				emo = blog['emo']
				if tf.has_key(emo):
					tf[emo] += 1
				else:
					tf[emo] = 1

	cPickle.dump(tf, open('data/blogs/emotf.pkl', 'w'))	
	
	top_emo = sorted(tf.items(), key = lambda k:-k[1])
	emos = [emo for emo, tf in top_emo[:500]]
	open('data/blogs/eid.txt', 'w').write(u'\n'.join(emos))

def split(eids):
	all_emos = open('data/blogs/eid.txt', 'r').read().decode('utf8').split('\n')

	eidmap = {}
	for eid in eids:
		eidmap[all_emos[eid]] = eid

	odname = 'data/blogs/eid_data/'
	if not os.path.isdir(odname):
		os.mkdir(odname)

	datalist = {}
	for eid in eids:
		datalist[eid] = []

	for i in range(3):
		fname = 'data/blogs/teaf/%d.txt'%(i)
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

def init_folders(dnames):
	for dname in dnames:
		if not os.path.isdir(dname):
			os.mkdir(dname)

def prepare_dataset(dname_dataset, n_emo, n_samples):
	idname = 'data/blogs/eid_data/'

	odname = 'data/blogs/%s/raw'%(dname_dataset)

	init_folders('data/blogs/%'%(dname_dataset), odname)

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

def main():
	#ifname = 'data/blogs/0s.txt'
	#ofname = 'data/blogs/out0s.txt'
	
	ifname = sys.argv[1]
	ofname = sys.argv[2]

	extract(ifname, ofname)

if __name__ == '__main__':
	main()

