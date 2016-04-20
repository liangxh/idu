#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys
reload(sys)
sys.setdefaultencoding('utf8')

import time
import cPickle
import numpy as np
from optparse import OptionParser


import datica
from utils import progbar

def dataset_to_texts(dataset):
	train, valid, test = dataset
	texts = []
	texts.extend(train[0])
	texts.extend(valid[0])
	texts.extend(test[0])
	return texts

def filter_valid_tokens(texts, rate):
	tf = {}

	for tokens in texts:
		for token in tokens:
			if tf.has_key(token):
				tf[token] += 1
			else:
				tf[token] = 1

	n_samples = len(texts)
	n_tokens = int(len(tf) * rate)

	tf = sorted(tf.items(), key = lambda k: -k[1])[: n_tokens]

	tokens_valid = {}
	for i, item in enumerate(tf):
		t, f = item
		tokens_valid[t] = i

	return tokens_valid

def get_valid_rate(texts, tokens_valid):
	c = 0
	valid_tokens = set(tokens_valid.keys())

	for tokens in texts:
		tokens = set(tokens)
		if len(tokens.intersection(valid_tokens)) == 0:
			c += 1

	print 'invalid coverage: %d (%.2f%%)'%(c, 100. * c / len(texts))

def f_sqrloss(x, y, w):
	return np.sum((y - np.dot(x, w)) ** 2)

def fgrad_sqrloss(x, y, w):
	return - np.sum( (y - np.dot(x, w)).reshape(x.shape[0], 1) * x, axis = 0)

class CKClassifier:	
	@classmethod
	def prepare_PMI(self, texts, tokens_valid, thr):
		n_tokens = len(tokens_valid)
		n_samples = len(texts)

		p_margin = np.zeros(n_tokens)
		p = np.zeros((n_tokens, n_tokens))
		
		pbar = progbar.start(n_samples)
		l = 0

		for tokens in texts:
			tids = [tokens_valid[token] for token in set(tokens) if tokens_valid.has_key(token)]

			for tid in tids:
				p_margin[tid] += 1

			for i in range(len(tids) - 1):
				t1 = tids[i]
				for j in range(i + 1, len(tids)):
					t2 = tids[j]
					if t1 < t2:
						p[t1][t2] += 1
					else:
						p[t2][t1] += 1
			
			l += 1
			pbar.update(l)
		pbar.finish()
		
		pmi_list = []
		#values = []
	
		n = (n_tokens - 1) * n_tokens / 2
		pbar = progbar.start(n)
		l = 0

		for i in range(n_tokens - 1):
			if p_margin[i] == 0.:
					print i
				
			for j in range(i + 1, n_tokens):
				v = p[i][j] / (p_margin[i] * p_margin[j])

				if v > thr:
					pmi_list.append(((i, j), v))
				
				#values.append(v)
				l += 1
				pbar.update(l)
		pbar.finish()

		#cPickle.dump(values, open('output/pmi_values.pkl', 'w'))
		return pmi_list

	@classmethod
	def prepare_sentiscore(self, train_x_y, tokens_valid, alpha, thr):
		x, y = train_x_y
		ydim = np.max(y) + 1
		n_tokens = len(tokens_valid)

		count = np.zeros((n_tokens, ydim))

		for tokens, y in zip(x, y):
			for token in tokens:
				if tokens_valid.has_key(token):
					tid = tokens_valid[token]
					count[tid][y] += 1

		count_sum = np.sum(count, axis = 0)
		scores = np.log2(((count[:, 0] + alpha) / (count_sum[0] + n_tokens * alpha) ) / ((count[:, 1] + alpha) / (count_sum[1] + n_tokens * alpha)))

		#cPickle.dump(sentiscore.tolist(), open('output/senti_values.pkl', 'w'))
		sentiscores = []
		
		thr_up = abs(thr)
		thr_down = -thr_up

		for i in range(scores.shape[0]):
			score = scores[i]
			if score >= thr_up:
				sentiscores.append(1)
			elif score <= thr_down:
				sentiscores.append(-1)
			else:
				sentiscores.append(0)

		return np.asarray(sentiscores)

	@classmethod
	def prepare_x(self, texts, tokens_valid):
		n_tokens = len(tokens_valid)
		n_samples = len(texts)
			
		x = np.zeros((n_samples, n_tokens))

		for i, tokens in enumerate(texts):
			for token in tokens:
				if tokens_valid.has_key(token):
					tid = tokens_valid[token]
					x[i][tid] += 1

		x /= np.sum(x, axis = 1).reshape(x.shape[0], 1)
		x[x == np.inf] = 0.
		return x

	@classmethod
	def prepare_y(self, ori_y):
		y = np.zeros(len(ori_y))
		for i, yi in enumerate(ori_y):
			y[i] = 1 if yi == 0 else -1

		return y

	
	def train(self, x, y, sim_tids, sentiscores):
		# rename
		A0 = []
		A1 = []
		for t1, t2 in sim_tids:
			A0.append(t1)
			A1.append(t2)

		Np = len(sim_tids)
		p = sentiscores
	
		# set function
		f = f_sqrloss
		f_grad = fgrad_sqrloss

		def calculate_v(w):
			return w[A0] - w[A1]

		def calculate_u(miu, rho):
			return miu / rho

		def calculate_turnAdot(w):
			r = np.zeros(xdim)
			for i in range(Np):
				a0 = A0[i]
				a1 = A1[i]	
				r[a0] += w[i]
				r[a1] -= w[i]
			return r

		def f_pos(x):
			return x if x > 0 else 0.

		def f_thresholding(a, k):
			y = np.zeros(a.shape[0])
			flags = (a - k > 0)
			y[flags] = a[flags]

			flags = (- a - k > 0)
			y[flags] -= a[flags]

			return y

		def g(z, f, x, y, v, u, p, alpha, lambda1, rho):
			return (f(x, y, z) - alpha * np.dot(p, z) 
					+ lambda1 * np.sum(z ** 2)
					+ rho / 2 * np.sum((calculate_v(z) - v + u) ** 2)
				)

		def g_grad(z, f_grad, x, y, v, u, p, alpha, lambda1, rho):
			return (f_grad(x, y, z) + alpha * p + 2 * lambda1 * z
					+ rho * calculate_turnAdot(calculate_v(z) - v + u)
				)

		def update_w(f, f_grad, x, y, w, v, u, p, alpha, lambda1, lambda2, rho, eta, L0):
			
			def sub_condition(z, s, L, f, f_grad, x, y, v, u, p, alpha, lambda1, rho):
				lhs = g(z, f, x, y, v, u, p, alpha, lambda1, rho)
				rhs = (
					g(s, f, x, y, v, u, p, alpha, lambda1, rho) + 
					np.dot(g_grad(s, f, x, y, v, u, p, alpha, lambda1, rho), (z - s)) + 
						L / 2 * np.sum((z - s) ** 2)
					)

				return lhs <= rhs

			def calculate_cost(f, x, y, w, v, u, p, alpha, lambda1, lambda2, rho):
				return (
					f(x, y, w) - alpha * np.dot(p, w)
					+ lambda1 * np.sum(w ** 2)	
					+ lambda2 * np.norm(w, 1)
					+ rho / 2 * np.sum((calculate_v(w) - v + u) ** 2)
					)
	
			z_f1 = w    # z_(t+1)
			z_b1 = 0    # z_(t-1)
			z = w

			k = 0.
			L = L0

			patience_max = 10
			patience_count = 0
			min_value = None
			max_epoch = 500000
			
			while True:
				z_b1 = z
				z = z_f1

				k += 1
				a = k / (k + 3)
				s_f1 = z + a * (z - z_b1)
				g_grad_s_f1 = g_grad(s_f1, f_grad, x, y, v, u, p, alpha, lambda1, rho)
			
				z_f1 = f_thresholding(s_f1 - g_grad_s_f1 / L, lambda2 / L)

				while not sub_condition(z_f1, s_f1, L, f, f_grad, x, y, v, u, p, alpha, lambda1, rho):
					L *= eta
					z_f1 = f_thresholding(s_f1 - g_grad_s_f1 / L, lambda2 / L)

				cost = calculate_cost(f, x, y, w, v, u, p, alpha, lambda1, lambda2, rho)

				if cost < min_value:
					min_value = cost
					patience_count = 0
				else:
					patience_count += 1
					if patience_count > patience_max:
						#print >> sys.stderr, 'update_w: [info] '
						break

				if k > max_epoch:
					#print >> sys.stderr, 'max_epoch met'
					break
			
			return z_f1

		def update_v(w, u, beta, rho):
			return f_thresholding(calculate_v(w) + u, beta / rho)

		def update_u(w, u, v):
			return u + calculate_v(w) - v
	
		# initialization
		xdim = x.shape[1]
	
		lambda1 = .5
		lambda2 = .5
		alpha = .5
		beta = .5
		rho = .5
		eta = 1.2
		L0 = 0.1

		w = np.random.randn(xdim)
		v = calculate_v(w)

		miu = np.random.randn(Np)
		u = calculate_u(miu, rho)

		def calculate_loss(f, x, y, w, p, alpha, beta, lambda1, lambda2):
			return (
				f(x, y, w) - alpha * np.dot(p, w) + beta * np.norm(calculate(w), 1)
				+ lambda1 * np.sum(w ** 2)
				+ lambda2 * np.norm(w, 1)
				)

		patience_max = 10
		patience_count = 0
		min_value = None

		max_epoch = 500000
		l = 0
		while True:
			print >> sys.stderr, 'updating w...'
			w = update_w(f, f_grad, x, y, w, v, u, p, alpha, lambda1, lambda2, rho, eta, L0)
			
			print >> sys.stderr, 'updating v...'
			v = update_v(w, u, beta, rho)

			print >> sys.stderr, 'updating u...'
			u = update_u(w, u, v)

			l += 1

			loss = calculate_loss(f, x, y, w, p, alpha, beta, lambda1, lambda2)
			print >> sys.stderr, 'EPOCH %d loss %f'%(loss)

			if loss < min_value:
				min_value = loss
				patience_count = 0
			else:
				patience_count += 1
				if patience_count > patience_max:
					print >> sys.stderr, 'train: [info] patience max met'
					break

			if l > max_epoch:
				print >> sys.stderr, 'train: [info] epoch max met'
				break

		self.w = w

def main():
	'''
	optparser = OptionParser()
	optparser.add_option('-k', '--keep_rate', action='store', dest='keep_rate', type='float', default = 0.2)
	opts, args = optparser.parse_args()

	###################### Load dataset #####################################
	config = datica.load_config('data/config2.txt')
	dataset = datica.load_by_config('data/dataset/unigram/', config, valid_rate = 0.)

	###################### Preparation ######################################
	train, valid, test = dataset
	texts = dataset_to_texts(dataset)

	rate_TF = opts.keep_rate
	thr_PMI = 0.02

	alpha_SENTI = 1.0
	thr_SENTI = 2.0

	tokens_valid = filter_valid_tokens(texts, rate_TF)
	get_valid_rate(texts, tokens_valid)
	
	##################### Prepare PMI #########################################
	pmi_list = CKClassifier.prepare_PMI(texts, tokens_valid, thr_PMI)
	#pmi_list is too long so thr_filtering is done within it

	sim_tids = [tid_pair for tid_pair, v in pmi_list]

	##################### Prepare SentiScore ##################################
	sentiscores = CKClassifier.prepare_sentiscore(train, tokens_valid, alpha_SENTI, thr_SENTI)
	
	x = CKClassifier.prepare_x(train[0], tokens_valid)
	y = CKClassifier.prepare_y(train[1])

	cPickle.dump((x, y, sim_tids, sentiscores), open('data/ckdata.pkl', 'w'))
	'''

	print >> sys.stderr, 'loading data...', 
	st = time.time()
	x, y, sim_tids, sentiscores = cPickle.load(open('data/ckdata.pkl', 'r'))
	print >> sys.stderr, ' done (%.2f sec)'%(time.time() - st)


	classifier = CKClassifier()
	classifier.train(x, y, sim_tids, sentiscores)
	
if __name__ == '__main__':
	main()

	
