#! /usr/env/python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.04.25
'''

import sys
reload(sys)
sys.setdefaultencoding('utf8')
import cPickle
import gensim
from optparse import OptionParser

import zhtokenizer
from utils import progbar, zhprocessor

def text_iterator():
	import db
	con = db.connect()
	cur = con.cursor()

	cur.execute('SELECT COUNT(*) FROM microblogs')
	n_text = cur.fetchone()[0]
	print >> sys.stderr, 'Totally %d text, executing SELECT text FROM microblogs...'%(n_text), 
	
	cur.execute('SELECT text FROM microblogs')
	print >> sys.stderr, 'OK'

	pbar = progbar.start(n_text)
	l = 0
	for t0 in cur:
		try:
			t = t[0].decode('utf8')
			t = zhprocessor.simplify(t)
		except:
			l += 1
			pbar.update(l)
			continue

		yield t
		
		l += 1
		pbar.update(l)
		
	pbar.finish()
	cur.close()
	con.close()

class DBTextIterator:
	def __init__(self):
		import db
		self.con = db.connect()
		self.cur = self.con.cursor()

		self.cur.execute('SELECT COUNT(*) FROM microblogs')
		n_text = self.cur.fetchone()[0]
		print >> sys.stderr, 'Totally %d text, executing SELECT text FROM microblogs...'%(n_text), 
	
		self.cur.execute('SELECT text FROM microblogs LIMIT 3')
		print >> sys.stderr, 'OK'

		self.pbar = progbar.start(n_text)
		self.l = 0

	def next(self):
		res = self.cur.fetchone()
		
		while res is not None:
			self.l += 1
			self.pbar.update(self.l)
			
			try:
				t = res[0]
				t = t.decode('utf8')
				t = zhprocessor.simplify(t).decode('utf8')
				#tokens = zhtokenizer.unigramize()
				#return tokens
				return t
			except:
				res = self.cur.fetchone()

		raise StopIteration

	def close(self):
		self.pbar.finish()
		self.cur.close()
		self.con.close()

	def __iter__(self):	
		return self

def main():
	optparser = OptionParser()
	optparser.add_option('-d', '--dim_proj', action='store', type = 'int', dest='dim_proj')
	optparser.add_option('-w', '--worker', action='store', type = 'int', dest='n_worker', default = 1)
	optparser.add_option('-o', '--output', action='store', type = 'str', dest='output')
	opts, args = optparser.parse_args()

	dbiter = DBTextIterator()

	m = gensim.models.Word2Vec(
		dbiter,
		size = opts.dim_proj,
		workers = opts.n_worker,
		)

	dbiter.close()

	m.save_word2vec_format(opts.output, binary = True)
	m = gensim.models.Word2Vec.load_word2vec_format(opts.output, binary = True)
	print m[u'我']

if __name__ == '__main__':
	main()
