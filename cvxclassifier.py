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
import cvxpy

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
	return - np.sum( (y - np.dot(x, w)).reshape(y.shape[0], 1) * x, axis = 0)

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

		vs = []
		for i in range(n_tokens - 1):
			if p_margin[i] == 0.:
					print i
				
			for j in range(i + 1, n_tokens):
				v = p[i][j] / (p_margin[i] * p_margin[j])
				vs.append(v)

				if v > thr:
					pmi_list.append(((i, j), v))
				
				#values.append(v)
				l += 1
				pbar.update(l)
		pbar.finish()

		print 'sim_value_range: [%f, %f]'%(np.min(vs), np.max(vs))

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
	def prepare(self, texts, labels, tokens_valid):
		n_tokens = len(tokens_valid)
		n_samples = len(texts)
		
		x = []
		y = []

		for i, tokens in enumerate(texts):
			vec = np.zeros(n_tokens)

			for token in tokens:
				if tokens_valid.has_key(token):
					tid = tokens_valid[token]
					vec[tid] += 1

			vec_sum = np.sum(vec)
			if not vec_sum == 0:
				vec /= vec_sum

			x.append(vec / vec_sum)
			y.append(1 if labels[i] == 0 else -1)

		return np.asarray(x), np.asarray(y)
	
	def train(self, x, y, p, A, alpha, beta, lambda1, lambda2):
		w = cvxpy.Variable(x.shape[1])
		objective = cvxpy.Minimize(
						cvxpy.sum_squares(y - x * w)
						- alpha * p * w
						+ beta * cvxpy.norm(A * w , 1)
						+ lambda1 * cvxpy.norm(w, 2) ** 2
						+ lambda2 * cvxpy.norm(w, 1)
					)

		prob = cvxpy.Problem(objective)
		result = prob.solve()

		self.w = w.value

	def classify(self, x):
		score_y = np.asarray(np.dot(x, self.w)).flatten()
		y = np.ones(x.shape[0])
		y[score_y < 0] = -1
		
		return y

def main():
	optparser = OptionParser()
	optparser.add_option('-k', '--keep_rate', action='store', dest='keep_rate', type='float', default = 0.2)
	optparser.add_option('-c', '--fname_config', action='store', type = 'str', dest='fname_config')
	optparser.add_option('-a', '--alpha', action='store', type = 'float', dest='alpha', default = 1.)
	optparser.add_option('-b', '--beta', action='store', type = 'float', dest='beta', default = 1.)
	optparser.add_option('-l', '--lambda1', action='store', type = 'float', dest='lambda1', default = 1.)
	optparser.add_option('-t', '--lambda2', action='store', type = 'float', dest='lambda2', default = 1.)
	
	opts, args = optparser.parse_args()

	alpha = opts.alpha
	beta = opts.beta
	lambda1 = opts.lambda1
	lambda2 = opts.lambda2

	print 'alpha: ', alpha
	print 'beta: ', beta
	print 'lambda1: ', lambda1
	print 'lambda2: ', lambda2

	###################### Load dataset #####################################

	config = datica.load_config(opts.fname_config)
	dataset = datica.load_by_config('data/dataset/unigram/', config, valid_rate = 0.)

	###################### Preparation ######################################
	train, valid, test = dataset
	texts = dataset_to_texts(dataset)

	rate_TF = opts.keep_rate
	thr_PMI = 0.003

	alpha_SENTI = 1.0
	thr_SENTI = 2.0

	tokens_valid = filter_valid_tokens(texts, rate_TF)
	get_valid_rate(texts, tokens_valid)

	##################### Prepare SentiScore ##################################
	p = CKClassifier.prepare_sentiscore(train, tokens_valid, alpha_SENTI, thr_SENTI)
	
	##################### Prepare PMI #########################################
	pmi_list = CKClassifier.prepare_PMI(texts, tokens_valid, thr_PMI)
	#pmi_list is too long so thr_filtering is done within it

	Np = len(pmi_list)
	print 'Np: %d'%(Np)

	t0 = []
	t1 = []
	for tid_pair, v in pmi_list:
 		t0.append(tid_pair[0])
		t1.append(tid_pair[1])
	r = range(Np)

	A = np.zeros((Np, len(tokens_valid)))
	A[r, t0] = 1
	A[r, t1] = -1
	
	x, y = CKClassifier.prepare(train[0], train[1], tokens_valid)

	#x, y, p, A = cPickle.load(open('cvxdata.pkl', 'r'))
	#print type(p)
	#print p.shape
	#return
	
	classifier = CKClassifier()
	classifier.train(x, y, p, A, alpha, beta, lambda1, lambda2)
	
	x, y = CKClassifier.prepare(test[0], test[1], tokens_valid)

	pred_y = classifier.classify(x)

	print len(pred_y[pred_y == 1])
	print len(pred_y[pred_y == -1])


	prec = 100. * len(pred_y[pred_y == y]) / pred_y.shape[0]
	print 'precision: %.2f%%'%(prec)	

if __name__ == '__main__':
	main()

	
