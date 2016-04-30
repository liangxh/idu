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
import theano
import numpy as np
from optparse import OptionParser


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

class VocabIterator:
	def __init__(self, n_emo):
		self.n_emo = n_emo

	def __iter__(self):
		pbar = progbar.start(self.n_emo)
		for i in range(self.n_emo):
			seqs = cPickle.load(open('data/dataset/unigram/%d.pkl'%(i), 'r'))
			for seq in seqs:
				yield seq
			pbar.update(i + 1)
		pbar.finish()

def to_Wemb(ifname, prefix, dim):
	#optparser = OptionParser()
	#optparser.add_option('-d', '--dim', action='store', type = 'int', dest='dim')
	#optparser.add_option('-i', '--input', action='store', type = 'str', dest='ifname')
	#optparser.add_option('-p', '--prefix', action='store', type = 'str', dest='prefix')
	#opts, args = optparser.parse_args()

	#ifname = 'data/word2vec_model_32.bin'

	print >> sys.stderr, 'loading model from %s ... '%(ifname),
	m = gensim.models.Word2Vec.load_word2vec_format(opts.ifname, binary = True)
	print >> sys.stderr, 'OK'

	n_emo = 90
	tokens = set()
	for eid in range(n_emo):
		seqs = cPickle.load(open('data/dataset/unigram/%d.pkl'%(eid), 'r'))
		for seq in seqs:
			tokens |= set(seq)

	Widx = {}
	vecs = [[0. for i in range(dim)], ]
	for token in tokens:
		try:
			vec = model[token]
		except KeyError:
			continue
		
		c += 1
		Widx[token] = c
		vecs.append(vec)

	Wemb = np.asarray(vecs).astype(theano.config.floatX)
	embedder = WordEmbedder(Widx, Wemb)
	
	fname_embedder = DIR_MODEL + '%s_embedder.pkl'%(prefix)	
	print >> sys.stderr, 'exporting embedder to %s ... '%(fname_embedder),
	embedder.dump(fname_embedder)
	print >> sys.stderr, 'OK'

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


	m.build_vocab(VocabIterator(90))
	m.train(DBTextIterator(50000000))

	m.save_word2vec_format(opts.output, binary = True)
	
if __name__ == '__main__':
	#main()
