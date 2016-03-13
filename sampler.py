#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.02.04
Modified: 2016.02.04
Description: filter text for certain emoticon
'''

import os
import sys
reload(sys)
sys.setdefaultencoding('utf8')

import cPickle
from const import DIR_EID_UIDS

def muid2umids(mid_uid):
	'''
	transfer {mid: uid, ...} to {uid: [mid, mid, ...], ...}
	'''

	uid_mids = {}

	for mid, uid in mid_uid.items():
		if uid_mids.has_key(uid):
			uid_mids[uid].append(mid)
		else:
			uid_mids[uid] = [mid, ]

	return uid_mids

def load_mid_uid(eid):
	'''
	load {mid:uid, ...} from data/eid_uids/$EID.pkl
	'''

	fname = DIR_EID_UIDS + '%d.pkl'%(eid)
	
	if not os.path.exists(fname):
		print '[Warning] %s not found'%(fname)
		return None

	return cPickle.load(open(fname, 'r'))

def sample(eid):
	'''
	sampling text from MySQL to data/dataset/text/EID.txt and data/dataset/muid/EID.txt
	'''

	import blogger
	import db
	con = db.connect()
	cur = con.cursor()

	mid_uid = load_mid_uid(eid)
	umids = muid2umids(mid_uid)
	
	uid_mids = sorted(umids.items(), key = lambda k: -len(k[1]))

	n_text = 0
	target = 4000

	texts = []
	str_mid_uid = []

	for i, item in enumerate(uid_mids):
		uid, mids = item
		
		c = 0
		for mid in mids:
			cur.execute('SELECT text FROM microblogs WHERE user_id=%s AND mid=%s LIMIT 1'%(uid, mid))
			text = cur.fetchone()[0]
			if not blogger.is_valid(text):
				continue

			texts.append(text)
			str_mid_uid.append('%s %s'%(mid, uid))
			c += 1
			if c == 100:
				break

		n_text += c
		if n_text >= target:
			break
	
	# n_text may not be equal to 4000, later correction - lxh

	if len(texts) > target:
		texts = texts[:target]
		str_mid_uid = str_mid_uid[:target]

	DIR_DATASET = 'data/dataset/'
	DIR_TEXT = DIR_DATASET + 'text/'
	DIR_MUIDS = DIR_DATASET + 'muid/'

	open(DIR_TEXT + '%d.txt'%(eid), 'w').write('\n'.join(texts))
	open(DIR_MUIDS + '%d.txt'%(eid), 'w').write('\n'.join(str_mid_uid))

if __name__ == '__main__':	
	import sys
	if len(sys.argv) < 2:
		eid = 0
	else:
		eid = int(sys.argv[1])	

	sample(eid)

	#analyse_mid_uid(0)
