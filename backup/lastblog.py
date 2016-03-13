#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.02.26
Description: get blogs from MySQL
'''

import sys
reload(sys)
sys.setdefaultencoding('utf8')

import time
import datetime

def exists(mid, threshold = 1, con = None):
	import db
	if con == None:
		conn = db.connect()
		cur = conn.cursor()

	cur.execute('SELECT user_id, timestamp FROM microblogs WHERE mid = %s LIMIT 1'%(mid))
	res = cur.fetchone()

	if res == None:
		print 'lastblog: [warning] blogs of MID %s not found'%(mid)
		res = False
	else:
		uid, post_time = res
		time_str = datetime.datetime.strftime(post_time, '%Y-%m-%d %H:%M:%S')

		cur.execute('SELECT mid FROM microblogs WHERE user_id = %s and timestamp < "%s" LIMIT %d'%(
				uid, time_str, threshold
			))
		mids = cur.fetchall()
		res = not ((mids is None) or (len(mids) < threshold))

	cur.close()
	
	if con == None:
		conn.close()

	return res

def get(mid, con = None):
	import db
	if con == None:
		conn = db.connect()
		cur = conn.cursor()

	cur.execute('SELECT user_id, timestamp FROM microblogs WHERE mid = %s LIMIT 1'%(mid))
	res = cur.fetchone()

	if res == None:
		print 'lastblog: [warning] blogs of MID %s not found'%(mid)
		res = None
	else:
		uid, post_time = res
		time_str = datetime.datetime.strftime(post_time, '%Y-%m-%d %H:%M:%S')

		cur.execute('SELECT mid, timestamp FROM microblogs WHERE user_id = %s and timestamp < "%s" ORDER BY timestamp DESC'%(uid, time_str))
		res = cur.fetchall()

	cur.close()
	
	if con == None:
		conn.close()

	return res

if __name__ == '__main__':

	if len(sys.argv) < 2:
		mid = '1090813127'
	else:
		mid = sys.argv[1]

	flag = exists(mid)
	if not flag:
		print 'no blogs found'
	else:
		res = get(mid)
		if res == None:
			pass
		else:
			print '%s blogs found'%(len(res))
			for i, item in enumerate(res):
				mid, post_time = item
				print '%d. %s %s'%(i + 1, mid, datetime.datetime.strftime(post_time, '%Y-%m-%d %H:%M:%S'))


