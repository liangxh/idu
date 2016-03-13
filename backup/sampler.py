#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.01.29
Modified: 2016.02.02
Description: sampling and filtering blogs in the database
'''

#import db

import cPickle
from share import emotica
from const import TXT_STATUS_COUNT, PKL_REPORT
from utils import timer, progbar

def tohist(ls):
	'''
	transfer {1, 1, 2, 3, 1, 2, 3} to {1:3, 2:2, 3:3}
	'''
	hist = {}
	for l in ls:
		if not hist.has_key(l):
			hist[l] = 1
		else:
			hist[l] += 1

	return hist

def tohlist(hist):
	'''
	transfer {'A':1, 'B':1, 'C':2} to {1:['A', 'B'], 2:['C', ]}
	'''
	hlist = {}
	for k, v in hist.items():
		if hlist.has_key(v):
			hlist[v].append(k)
		else:
			hlist[v] = [k, ]

	return hlist

def get_status_count():
	'''
	load the dict of {UID: N_BLOGS, ...}
	'''

	fobj = open(TXT_STATUS_COUNT, 'r')
	lines = fobj.read().split('\n')
	fobj.close()

	status_count = {}
	for line in lines:
		args = line.split(' ')
		status_count[args[0]] = int(args[1])

	return status_count

def analyse(uid):
	'''
	'''

	import db
	import datica

	con = db.connect()
	cur = con.cursor()
	cur.execute('SELECT text, comments_count FROM microblogs WHERE user_id=%s'%(uid))

	valid_count = 0
	comm_count = 0

	emoticons = []
	for text, comments_count in cur:
		res = datica.extract(text)
		if res == None:
			continue
		
		text, emo = res
		emoticons.append(emo)

		valid_count += 1
		if comments_count > 0:
			comm_count += 1

	emoticons = tohist(emoticons)
		
	return valid_count, comm_count, emoticons

def sampling():
	status_count = get_status_count()
	hlist_count = tohlist(status_count)

	sorted_items = sorted(hlist_count.items(), key = lambda k: - len(k[1]))
	
	sample_items = []
	sample_items.extend(sorted_items[1][1])

	valid_list = []
	comm_list = []
	emo_tf = {}
	emo_df = {}

	pbar = progbar.start(len(sample_items))

	for i, uid in enumerate(sample_items):
		valid_count, comm_count, emos = analyse(uid)
		
		valid_list.append(valid_count)
		comm_list.append(comm_count)
		
		for emo, count in emos.items():
			if emo_tf.has_key(emo):
				emo_tf[emo] += count
				emo_df[emo] += 1
			else:
				emo_tf[emo] = count
				emo_df[emo] = 1
		pbar.update(i + 1)
	pbar.finish()

	valid_hist = tohist(valid_list)
	comm_hist = tohist(comm_list)

	cPickle.dump((valid_hist, comm_hist, emo_tf, emo_df), open(PKL_REPORT, 'w'))

if __name__ == '__main__':
	sampling()

