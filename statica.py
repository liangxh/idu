#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.02.03
Modified: 2016.02.03
Description: create statistics report from mysql
'''

import cPickle
import os
import sys
reload(sys)
sys.setdefaultencoding('utf8')

import blogger
from const import TOTAL_BLOGS, PKL_EMO_MIDS, DIR_EID_MIDS, TXT_EID, DIR_EID_UIDS

def collect_emo_mids():
	'''
	collect {emo: [mid, mid, ..], } from mysql and export to PKL_EMO_MIDS
	'''

	import db
	from utils import probgar

	print 'connecting to MySQL..'
	con = db.connect()

	print 'start..'
	cur = con.cursor()
	cur.execute('SELECT mid, text FROM microblogs WHERE comments_count > 1')
	
	pbar = progbar.start(TOTAL_BLOGS)	
	loop = 0
	emo_mids = {}
	for mid, text in cur:
		res = blogger.extract(text)
		if res == None:
			continue
		
		text, emo = res
		if emo_mids.has_key(emo):
			emo_mids[emo].append(mid)
		else:
			emo_mids[emo] = [mid, ]
	
		loop += 1
		pbar.update(loop)

	pbar.finish()

	cPickle.dump(emo_mids, open(PKL_EMO_MIDS, 'w'))
	
	cur.close()
	con.close()

def analyse_emo_mids():
	'''
	show some statics about {emo: [mid, mid, ..], } from PKL_EMO_MIDS
	'''
	import matplotlib.pyplot as plt

	emo_mids = cPickle.load(open(PKL_EMO_MIDS, 'r'))
	emo_count = {}
	for k, v in emo_mids.items():
		emo_count[k] = len(v)

	n_blogs = sum(emo_count.values())	
	emo_count = sorted(emo_count.items(), key = lambda k:-k[1])

	coverage = []
	last = 0.
	for k, v in emo_count:
		last += 100. * v / n_blogs
		coverage.append(last)
	
	print '### Coverage: # of Emoticons ###'
	th = [90., 95., 97., ]
	th_id = 0
	for i, c in enumerate(coverage):
		if c > th[th_id]:
			print '%0.2f: %d'%(th[th_id], i + 1)
			th_id += 1
			if th_id >= len(th):
				break

	plt.figure()
	plt.xlabel('emoticons')
	plt.ylabel('coverage (%)')
	plt.plot(coverage)
	plt.savefig('output/coverage_of_emoticons.png')

def fname_eid_mids(eid):
	return DIR_EID_MIDS + '%d.txt'%(eid)

def split_emo_mids(n = 200):
	'''
	split PKL_EMO_MIDS to eid_mids/%d.txt of list of mids,  %d in [0, 199]
	'''

	emo_mids = cPickle.load(open(PKL_EMO_MIDS, 'r'))
	
	emo_count = {}
	for k, v in emo_mids.items():
		emo_count[k] = len(v)

	emo_count = sorted(emo_count.items(), key = lambda k:-k[1])

	open('data/eid.txt', 'w').write('\n'.join([k for k, v in emo_count[:n]]))

	if not os.path.exists(DIR_EID_MIDS):
		print 'Remind: mkdir data/eid_mids'
	else:
		for eid, item in enumerate(emo_count[:n]):
			emo, count = item
			mids = emo_mids[emo]
			open(fname_eid_mids(eid), 'w').write('\n'.join(mids))

def read_eid_mids(eid):
	fname = fname_eid_mids(eid)

	try:
		mids = open(fname, 'r').read().split('\n')
	except:
		mids = None
		print '[Warning] %s not found'%(fname)
	
	return	mids

def get_uids(mids, con):
	cur = con.cursor()

	mids = set(mids)
	n_mids = len(mids)

	cur.execute('SELECT user_id, mid FROM microblogs WHERE mid in (%s)'%(','.join(mids)))

	uids = {}
	c = 0
	for uid, mid in cur:
		if not mid in mids:
			continue 

		if uids.has_key(mid):
			print 'CRASH: ', mid
		else:
			uids[mid] = uid
			c += 1
			if c == n_mids:
				break

	cur.close()
	print '%0.2f%% (%d / %d) found'%(100.* c / n_mids, c, n_mids)

	return uids

def export_uids(eids):
	'''
	export uids = {mid: uid, } -> uids[mid] = uid to respective DIR_EID_UIDS/EID.pkl
	'''

	if not os.path.exists(DIR_EID_UIDS):
		print '[Remind] mkdir %s'%(DIR_EID_UIDS)

	import db
	con = db.connect()

	for eid in eids:
		print '## EID: %d'%(eid)
		mids = read_eid_mids(eid)

		uids = get_uids(mids, con)
		cPickle.dump(uids, open(DIR_EID_UIDS + '%d.pkl'%(eid), 'w'))

	con.close()

if __name__ == '__main__':
	#collect_emo_mids()
	#analyse_emo_mids()
	#split_emo_mids()

	export_uids(range(200))

