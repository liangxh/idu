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
#from rand_embedder import RandEmbedder

import wemb_rand
from wordembedder import WordEmbedder

class LstmScriptRand(LstmScript):
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
				for set_x, set_y in dataset:
					for x in set_x:
						yield x

			embedder = WordEmbedder(*wemb_rand.build(x_iterator(dataset), self.opts.dim_proj))
			embedder.dump(fname_embedder)
		
			return embedder

def main():
	script = LstmScriptRand()
	script.run()

if __name__ == '__main__':
	main()

