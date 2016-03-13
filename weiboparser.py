#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.03.03

'''

import sys
reload(sys)
sys.setdefaultencoding('utf8')
import time
from threading import Thread

import weibocomm
from utils.urlopener import UrlOpener

def load_accounts():
	fobj = open('data/weibo_account_cookie.txt', 'r')
	lines = fobj.readlines()

	i = 0
	idx = 0
	n_lines = len(lines)

	accounts = []
	while (i + 1 < n_lines):
		params = lines[i].strip().split(' ')
		uid = params[0]
		email = params[1]
		password = params[2]
		cookie = lines[i + 1].strip()

		accounts.append(WeiboAccount(uid, email, password, cookie))

		idx += 1
		i += 3

	return accounts

class WeiboAccount:
	def __init__(self, uid, email, password, cookie):
		self.uid = uid
		self.email = email
		self.password = password
		self.cookie = cookie

class WeiboParser(Thread):
	def __init__(self):
		Thread.__init__(self)

		self.urlopener = UrlOpener()
		self.account = None
		self.comm_uid = None
		self.comm_mid = None

	def set_account(self, account):
		self.account = account
		self.urlopener.set_cookie(account.cookie)

	def parse(self, uid, mid, show_result = False, show_max = 1000):
		if (self.account is None) or (self.urlopener.empty_cookie()):
			print 'weiboparser.parse: [warning] please set_account() before parse()'
			return None
		
		ret = weibocomm.get(self.urlopener, uid, mid, show_result, show_max)
		return ret

	def set_comm_info(self, uid, mid):
		self.comm_uid = uid
		self.comm_mid = mid

	def run(self):
		if self.comm_uid is None or self.comm_mid is None:
			print 'weiboparser.parse: [warning] please set_comm_info() before run()'
			return None

		ret = self.parse(self.comm_uid, self.comm_mid)

def test():
	uid, mid = ('1427605041', '3509858479270286')

	accounts = load_accounts()
	
	for ac in accounts[:3]:
		print ac.email
		parser = WeiboParser()
		parser.set_account(ac)
		parser.set_comm_info(uid, mid)
		parser.start()
	return

def test_one():
	import random

	#uid, mid = ('1427595804', '3521736785812544')
	uid, mid = ('1427816602 | 3436166978041857'.strip().split(' | '))	
	#uid, mid = ('1427616842	3521448876041091'.strip().split('\t'))

	accounts = load_accounts()
	ac = accounts[random.randint(0, len(accounts) - 1)]

	print 'user_id: %s'%(uid)
	print 'mid: %s'%(mid)
	print 'text: '
	print
	print 'comments_count (in MySL): '
	print

	ac.email
	parser = WeiboParser()
	parser.set_account(ac)
	comm, ids = parser.parse(uid, mid, show_result = True, show_max = 2000)
	print len(comm)

if __name__ == '__main__':
	#data = load()
	#print len(data)
	#print data[0]

	#test()
	test_one()

