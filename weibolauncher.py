#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.03.06
'''

import os
import sys
reload(sys)
sys.setdefaultencoding('utf8')
from optparse import OptionParser

import re
import json
import time
import threading


import commdatica
import weiboparser
from weiboparser import WeiboParser
from utils.urlopener import UrlOpener
from utils.logger import Logger

logger = Logger(sys.stderr)

JSONS_COMMENT = 'data/comments.txt'

def load(fname):
	'''
	load all the comments downloaded and saved in JSONS_COMMENT
	'''

	if not os.path.exists(fname):
		return []

	fobj = open(fname, 'r')
	comments = []
	for line in fobj:
		comments.append(json.loads(line))

	fobj.close()

	return comments

def downloaded_mids(fname):
	'''
	get the list of mids from JSONS_COMMENT
	which are the mids of the blogs whose comments have been downloaded
	'''

	comments = load(fname)

	mids = [] if comments == [] else [comm['mid'] for comm in comments]

	return mids

class ListIterator:
	'''
	an iterator for multi-threading
	'''

	def __init__(self, data, iter_round = False):
		self.data = data
		self.datalen = len(data)
		self.idx = 0
		self.iter_round = iter_round

		self.lock = threading.Lock()
	
	def restart(self):
		self.lock.acquire()
		self.idx = 0
		self.lock.release()

	def next(self):
		self.lock.acquire()

		if self.idx == self.datalen:
			if not self.iter_round:
				self.lock.release()
				return None, None
			else:
				self.idx = 0

		this_idx = self.idx
		datum = self.data[self.idx]
		self.idx += 1

		self.lock.release()

		return datum, this_idx

class SharedOutFile:
	'''
	a write file with threading lock though I don't know if it is necessary
	'''

	def __init__(self, fname, ftype = 'w'):
		self.fobj = open(fname, ftype)
		self.lock = threading.Lock()

	def write(self, line):
		self.lock.acquire()
		self.fobj.write(line)
		self.lock.release()

	def close(self):
		self.fobj.close()

class WeiboLauncher:
	'''
	a launcher for parsing weibo comments by multi-threading
	'''

	def __init__(self):
		pass

	def load_accounts(self, accounts):
		'''
		initialize a round iterator for accounts
		'''
		self.iter_account = ListIterator(accounts, iter_round = True)

	def load_bloginfo(self, bloginfo):
		'''
		initialize a iterator for bloginfos
		'''
		self.iter_bloginfo = ListIterator(bloginfo)

	def init_outfile(self, fname, ftype):
		'''
		initialize a output file object for threading
		'''
		self.outfile = SharedOutFile(fname, ftype)

	def close_outfile(self):
		'''
		close the output file
		'''
		if self.outfile is not None:
			self.outfile.close()

	def thread_parse(self, interval, thread_name = None):
		'''
		an instance for parseing weibo comments
		which sleep for $interval seconds between every loop
		'''

		flag = True

		wbparser = WeiboParser()

		bloginfo, bidx = self.iter_bloginfo.next()
		account, _ = self.iter_account.next()

		flag = not (bloginfo is None or account is None)

		fail_count = 0

		while flag and not self.thread_interrupt:
			#weibolauncher.thread_parse: 
			logger.info('<BLOG %d>: thread_%s downloading (%s %s)'%(
					bidx, thread_name, bloginfo.uid, bloginfo.mid))

			wbparser.set_account(account)
			ret = wbparser.parse(bloginfo.uid, bloginfo.mid)
			
			if ret == None:
				logger.info('<BLOG %d>: thread_%s failed'%(bidx, thread_name))
				
				fail_count += 1
				if fail_count > 3:
					print 'fail > 3'
					break
			else:
				comm, ids = ret
				blog = {}
				blog.update(bloginfo.todict())
				blog['comments_count'] = len(comm)
				blog['comments'] = comm
				blog['ids'] = ids
							
				self.outfile.write(json.dumps(blog).replace('\n', ' ') + '\n')
			
			if self.thread_interrupt:
				logger.info('thread_%s interrupted'%(thread_name))
				break

			bloginfo, bidx = self.iter_bloginfo.next()
			account, _ = self.iter_account.next()
			
			flag = not (bloginfo is None or account is None)

			if flag:
				time.sleep(interval)

	def launch(self, n_instance, interval):
		'''
		parse weibo comments by $n_instance threads
		accounts and bloginfo must be initialized by init_account and init_bloginfo in advance
		'''

		intr_lock = threading.Lock()
		self.thread_interrupt = False
		threads = []

		for i in range(n_instance):
			thread = threading.Thread(target = self.thread_parse, args = (interval, i, ))
			thread.start()

			threads.append(thread)
		
		try:
			for thread in threads:
				thread.join()
		except KeyboardInterrupt:
			self.thread_interrupt = True
			print 'after keyboardinterrupt'

			for thread in threads:
				thread.join()
			

def test():
	all_accounts = weiboparser.load_accounts()
	accounts = all_accounts[:25]

	all_bloginfo = commdatica.load()
	
	# do not download comments for the same blog again
	mids = set(downloaded_mids())
	filtered_bloginfo = [bloginfo for bloginfo in all_bloginfo if not bloginfo.mid in mids]

	bloginfo = filtered_bloginfo[:8]

	launch(JSONS_COMMENT, accounts, bloginfo, 4, 'w')


def launch(outfile, accounts, bloginfo, ftype, n_instance, interval):
	launcher = WeiboLauncher()
	launcher.load_accounts(accounts)
	launcher.load_bloginfo(bloginfo)
	launcher.init_outfile(outfile, ftype)
	launcher.launch(n_instance, interval)
	launcher.close_outfile()

def main():
	# get parameters from terminal
	optparser = OptionParser()
	optparser.add_option('-i', '--input', action = 'store', type = 'string', dest = 'infile')
	optparser.add_option('-o', '--output', action = 'store', type = 'string', dest = 'outfile')
	optparser.add_option('-a', '--account', action = 'store', type = 'string', dest = 'acc_range')
	optparser.add_option('-n', '--instance', action = 'store', type = 'int', dest = 'n_instance', default = 5)
	optparser.add_option('-r', '--restart', action = 'store_true', dest = 'restart', default = False)
	optparser.add_option('-t', '--interval', action = 'store', type = 'int', dest = 'interval', default = 3)

	opts, args = optparser.parse_args()

	if not opts.infile:
		print '-i infile not specified'
		return 

	if not opts.outfile:
		print '-o outfile not specified'
		return

	if not opts.acc_range:
		print '-a (start_idx,end_idx) not specified'
		return
	else:
		m = re.match('(\d+),(\d+)', opts.acc_range)
		if not m:
			print '-a start_idx,end_idx should contain no space'
			return
		else:
			opts.acc_range = (int(m.group(1)), int(m.group(2)))

	ftype = 'w' if opts.restart else 'a'

	# prepare the accounts
	all_accounts = weiboparser.load_accounts()
	accounts = all_accounts[opts.acc_range[0]:opts.acc_range[1] + 1]

	# prepare the 
	all_bloginfo = commdatica.load(opts.infile)

	# filter the blogs whose comments have been downloaded

	if opts.restart:
		mids = set()
	else:
		mids = set(downloaded_mids(opts.outfile))
		logger.info('%d downloaded in %s'%(len(mids), opts.outfile))

	bloginfos = [bloginfo for bloginfo in all_bloginfo if not bloginfo.mid in mids]

	# for test
	# bloginfos = bloginfos[:20]

	launch(opts.outfile, accounts, bloginfos, ftype, opts.n_instance, opts.interval)

if __name__ == '__main__':
	#test()
	main()

