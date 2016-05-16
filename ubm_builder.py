#! /usr/env/python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.05.16
'''

import sys
reload(sys)
sys.setdefaultencoding('utf8')
import cPickle
import numpy as np
from optparse import OptionParser
import time
from sklearn.mixture import GMM

import zhtokenizer
from wordembedder import WordEmbedder
from utils import progbar, zhprocessor
from const import DIR_MODEL

class DBTextIterator:
	def __init__(self, limit = None):
		self.limit = limit

	def __iter__(self):	
		import db
		con = db.connect()
		cur = con.cursor()

		if self.limit is None:
			cur.execute('SELECT COUNT(*) FROM microblogs')
			n_text = cur.fetchone()[0]
			print >> sys.stderr, 'Totally %d text'%(n_text)
		else:
			n_text = self.limit 
	
		sql = 'SELECT text FROM microblogs' + (' LIMIT %d'%(self.limit) if self.limit is not None else '')
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
	optparser.add_option('-e', '--key_embedder', action='store', type = 'str', dest='key_embedder')
	opts, args = optparser.parse_args()

	fname_embedder = 'data/dataset/model/%s_embedder.pkl'%(opts.key_embedder)
	embedder = WordEmbedder.load(fname_embedder)

	print >> sys.stderr, 'ubm_builder: [info] preparing x'

	iterator = DBTextIterator(50000000)
	x = []
	for seq in iterator:
		x.append(np.mean(embedder.embed(seq), axis = 0))


	for n in [8, 4, 16, 32]:
		print >> sys.stderr, 'ubm_builder: [info] fitting model for n = %d ...'%(n),
		st = time.time()

		ubm = GMM(n_components = n)
		ubm.fit(x)

		print >> sys.stderr, ' OK (%.2f sec)'%(time.time() - st)

		cPickle.dump(ubm, open('data/dataset/gmmubm/db_%s_%d.pkl'%(opts.key_embedder, n), 'w'))

if __name__ == '__main__':
	main()
	
