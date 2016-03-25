#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.03.16
'''

import os
import sys
reload(sys)
sys.setdefaultencoding('utf8')

from lstmscript import LstmScript

import wemb_cooc
import dimreducer
from wordembedder import WordEmbedder

class LstmScriptSVD(LstmScript):
	def add_extra_options(self):
		self.optparser.add_option('-d', '--dim_proj', action='store', type = 'int', dest='dim_proj') # , default = 128

	def init_embedder(self, dataset, fname_embedder):
		'''
		initialize the embedder by load it from file if available
		or build the model by the dataset and save it
		'''

		if os.path.exists(fname_embedder):
			print >> sys.stderr, 'embedding model %s found and loaded'%(fname_embedder)
			return WordEmbedder.load(fname_embedder)
		else:
			def x_iterator(dataset):
				all_x = []
				for set_x, set_y in dataset:
					all_x.extend(set_x)
				return all_x

			print >> sys.stderr, 'initialization of wordembedder'
			embedder = WordEmbedder(*wemb_cooc.build(x_iterator(dataset)))

			print >> sys.stderr, 'performing dimension reduction (svd)'
			embedder.dimreduce_fn(dimreducer.svd, self.opts.dim_proj)
			embedder.dump(fname_embedder)
		
			return embedder

def main():
	script = LstmScriptSVD()
	script.run()

if __name__ == '__main__':
	main()

