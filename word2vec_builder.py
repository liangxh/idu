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
import traceback
from optparse import OptionParser

import zhtokenizer
from utils import progbar, zhprocessor

class DBTextIterator:
	def __init__(self, limit = None):
		self.limit = limit

	def __iter__(self):	
		import db
		con = db.connect()
		cur = con.cursor()

		if self.limit is not None:
			cur.execute('SELECT COUNT(*) FROM microblogs')
			n_text = cur.fetchone()[0]
			print >> sys.stderr, 'Totally %d text'%(n_text)
		else:
			n_text = self.limit 
	
		sql = 'SELECT text FROM microblogs' + ' LIMIT %d'%(self.limit) if self.limit is not None else ''
		print >> sys.stderr, 'executing "%s"'%(sql)
		cur.execute(sql)
		print >> sys.stderr, 'OK'

		pbar = progbar.start(n_text)
		l = 0
		for t0 in cur:
			try:
				t = t0[0].decode('utf8')
				t = zhprocessor.simplify(t)				
			except:
				print traceback.format_exc()
				l += 1
				pbar.update(l)
				continue

			tokens = zhtokenizer.unigramize(t)
			yield tokens
		
			l += 1
			pbar.update(l)
		
		pbar.finish()
		cur.close()
		con.close()

def main():
	optparser = OptionParser()
	optparser.add_option('-d', '--dim_proj', action='store', type = 'int', dest='dim_proj')
	optparser.add_option('-w', '--worker', action='store', type = 'int', dest='n_worker', default = 1)
	optparser.add_option('-o', '--output', action='store', type = 'str', dest='output')
	opts, args = optparser.parse_args()

	m = gensim.models.Word2Vec(
		size = opts.dim_proj,
		workers = opts.n_worker,
		min_count = 1,
		)
	m.build_vocab(DBTextIterator(10))
	m.train(DBTextIterator(10))

	m.save_word2vec_format(opts.output, binary = True)
	#m = gensim.models.Word2Vec.load_word2vec_format(opts.output, binary = True)
	#print m[u'我']

if __name__ == '__main__':
	main()
