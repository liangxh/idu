#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.03.14
Description: Word Embedding using Denoise Autoencoder
'''

import theano
import numpy as np
from utils import dAtool

def shared_dataset(data_x, borrow=True):
	'''
	reference to DeepLearningTutorials.logistic_sgd.load_data.shared_dataset
	'''
	return theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)

def train(
	train_set_x,
	n_hidden,

	# changed recommended
	corruption_level = 0.,
	training_epochs = 15,

	# params for training
	batch_size = 64,
	learning_rate = 0.1,

	# save
	saveto = None,
	):

	n_visible = len(train_set_x[0])
	train_set_x = shared_dataset(train_set_x)

	da = dAtool.train(
		train_set_x = train_set_x, 
		n_visible = 28 * 28,
		n_hidden = 500,
		training_epochs = training_epochs,
		corruption_level = 0.3,
	)

	if saveto is not None:
		da.dump(saveto)

	return da

class dAEmbedder:
	'''
	an denoise autocoder embedder based on trained dA model
	'''
	def __init__(self, da):
		self.da  = da

	@classmethod
	def load(self, src):
		if isinstance(src, dAtool.dA):
			'''
			module dA expected
			'''
			return dAEmbedder(src)
		elif isinstance(src, str):
			'''
			filename of saved model expected
			'''
			return dAEmbedder(dAtool.load_model(fname_model))
		else:
			print 'unexpected data type for dAEmbedder.load'
			return None

	def embed(self, x):
		return self.da.get_hidden_values(x).eval()

def test():
	try:
		import PIL.Image as Image
	except ImportError:
		import Image

	from utils.dautils import tile_raster_images
	
	def load_data(fname = 'data/mnist.pkl.gz'):
		import cPickle
		import gzip
		f = gzip.open(fname, 'rb')
		dataset = cPickle.load(f)
		f.close()

		return dataset

	#xs = dAtool.load_train_subset('data/dset.pkl')
	datasets = load_data()
	train_set_x, train_set_y = datasets[0]
	xs = train_set_x

	da = train(xs, 100, corruption_level = 0.3)
	
	ori = []
	rei = []
	for x in xs[:10]:
		geth = da.get_hidden_values(x).eval()
		geto = da.get_reconstructed_input(geth).eval()
		newo = np.asmatrix(geto).T
		
		ori.append(x)
		rei.append(newo)

	images = []
	images.extend(ori)
	images.extend(rei)
	images = np.asmatrix(images)

	image = Image.fromarray(
		tile_raster_images(
			X = images,
			img_shape = (28, 28),
			tile_shape = (10, 10),
			tile_spacing = (1, 1)
		)
	)

	image.save('output/compare03.png')

if __name__ == '__main__':
	test()
	
