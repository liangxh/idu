# -*- encoding:utf8 -*-

# Description: Lstm implemented by Theano
# Required: lstm.py
# How to Use: please read __main__ for command line options

import os
import sys
from lstm import *
from dsetreader import *
from optparse import OptionParser

import numpy as np

default_fname_model = 'lstm_model.npz'

def load_eid(fname_eid):
	fobj = open(fname_eid, 'r')
	eid = json.load(fobj)
	fobj.close()

	return eid

class ThLstm:
	def __init__(self):
		None

	def prepare_x(self, seqs, maxlen = None):
		lengths = [len(s) for s in seqs]
	
		if maxlen is not None:
			new_seqs = []
			new_lengths = []
			for l, s in zip(lengths, seqs):
				if l < maxlen:
					new_seqs.append(s)
					new_lengths.append(l)
			lengths = new_lengths
			seqs = new_seqs
		
			if len(lengths) < 1:
				return None, None
		
		n_samples = len(seqs)
		maxlen = np.max(lengths)

		x = np.zeros((maxlen, n_samples)).astype('int64')
		x_mask = np.zeros((maxlen, n_samples)).astype(theano.config.floatX)

		for idx, s in enumerate(seqs):
			x[:lengths[idx], idx] = s
			x_mask[:lengths[idx], idx] = 1.

		return x, x_mask

	def prepare_data(self, seqs, labels, maxlen = None):
		lengths = [len(s) for s in seqs]
	
		if maxlen is not None:
			new_seqs = []
			new_labels = []
			new_lengths = []
			for l, s, y in zip(lengths, seqs, labels):
				if l < maxlen:
					new_seqs.append(s)
					new_labels.append(y)
					new_lengths.append(l)
			lengths = new_lengths
			labels = new_labels
			seqs = new_seqs
		
			if len(lengths) < 1:
				return None, None, None
		
		n_samples = len(seqs)
		maxlen = np.max(lengths)

		x = np.zeros((maxlen, n_samples)).astype('int64')
		x_mask = np.zeros((maxlen, n_samples)).astype(theano.config.floatX)

		for idx, s in enumerate(seqs):
			x[:lengths[idx], idx] = s
			x_mask[:lengths[idx], idx] = 1.

		return x, x_mask, labels

	'''def get_data(self,
		valid_portion = 0.2,
		test_portion = 0.1,

		fname_data = default_fname_data
	):
		samples = datareader.readJsonLines(self.fname_data)

		eidx = 0
		self.eid = {}		

		col = {}
		for s in samples:
			emo = s['emo']
			if not self.eid.has_key(emo):
				self.eid[emo] = eidx
				eidx += 1

			if not col.has_key(emo):
				col[emo] = []
			col[emo].append(s)

		self.dim = len(self.eid.keys())

		train_set_x = []
		train_set_y = []
		valid_set_x = []
		valid_set_y = []
		test_set_x = []
		test_set_y = []

		for k, v in col.items():
			n_samples = len(v)
			sidx = np.random.permutation(n_samples)
			n_test = int(numpy.round(n_samples * test_portion))
			n_valid = int(numpy.round(n_samples * valid_portion))
			n_train = n_samples - n_test - n_valid

			train_set_x.extend([v[s]['text'] for s in sidx[:n_train]])
			train_set_y.extend([self.eid[k] for s in sidx[:n_train]])
 			
			valid_set_x.extend([v[s]['text'] for s in sidx[n_train:(n_train + n_valid)]])
			valid_set_y.extend([self.eid[k] for s in sidx[n_train:(n_train + n_valid)]])

			test_set_x.extend([v[s]['text'] for s in sidx[(n_train + n_valid):]])
			test_set_y.extend([self.eid[k] for s in sidx[(n_train + n_valid):]])

		train = (train_set_x, train_set_y)
		valid = (valid_set_x, valid_set_y)
		test = (test_set_x, test_set_y)	

		return train, valid, test'''

	def reload_model(
		self,
		ydim,
		n_words,
		dim_proj = 128,
		encoder = 'lstm',
		use_dropout = True,
		fname_model = 'lstm_model.npz',
	):
		model_options = locals().copy()
		params = init_params(model_options)
		
		load_params(fname_model, params)
		tparams = init_tparams(params)
		(use_noise, x, mask, y, 
			f_pred_prob, f_pred, cost) = build_model(tparams, model_options)

		self.f_pred = f_pred
		self.f_pred_prob = f_pred_prob

	def classify(self, xs, batch_size = 64, show_progress = False):
		kf = get_minibatches_idx(len(xs), batch_size)

		preds = []
		pred_probs = []

		if show_progress:
			progbar.show(len(xs))
			progbar.init_clock()

		for _, idx in kf:
			x, x_mask = self.prepare_x([xs[i] for i in idx])
			ps = self.f_pred(x, x_mask)
			pds = self.f_pred_prob(x, x_mask)
			preds.extend(ps)
			pred_probs.extend(pds)
			
			if show_progress:
				progbar.click(len(idx))

		if show_progress:
			progbar.end()

		#x, x_mask = self.prepare_x(xs)
		#preds = self.f_pred(x, x_mask)
		#pred_probs = self.f_pred_prob(x, x_mask)

		return preds, pred_probs

	def train(
		self,
		dataset,
		ydim,
		n_words,

		dim_proj = 128,
		encoder = 'lstm',
		saveto = 'lstm_model.npz',
		validFreq = 1000,
		saveFreq = 1000,

		patience = 10,
		max_epochs = 5000,
		dispFreq = 10,
		decay_c = 0.,
		lrate = 0.0001,
		batch_size = 16,
		valid_batch_size = 64,
		optimizer = adadelta,

		# Parameter for extra option
		noise_std = 0.,
		use_dropout = True,		
		reload_model = None, #True,
		test_size = -1
	):
		train, valid, test = dataset
		
		if test_size > 0:
			# get random test_set
			idx = np.random.shuffle.permutation(len(test[0]))
			idx = idx[:test_size]
			test = ([test[0][n] for n in idx], [test[1][n] for n in idx])
		
		print 'Building model'
		model_options = locals().copy()
		params = init_params(model_options)
		
		if reload_model:
			if os.path.exists(saveto):
				load_params(saveto, params)
			else:
				print 'Warning: Model %s not found'%(saveto)

		tparams = init_tparams(params)

		(use_noise, x, mask, y, 
			f_pred_prob, f_pred, cost) = build_model(tparams, model_options)

		if decay_c > 0.:
			decay_c = theano.shared(numpy_floatX(decay_c), name='decay_c')
			weight_decay = 0.
			weight_decay += (tparams['U'] ** 2).sum()
			weight_decay *= decay_c
			cost += weight_decay

		f_cost = theano.function([x, mask, y], cost, name='f_cost')

		grads = tensor.grad(cost, wrt=tparams.values())
		f_grad = theano.function([x, mask, y], grads, name='f_grad')

		lr = tensor.scalar(name='lr')
		f_grad_shared, f_update = optimizer(lr, tparams, grads,
						x, mask, y, cost)

		print 'Optimization'

		kf_valid = get_minibatches_idx(len(valid[0]), valid_batch_size)
		kf_test = get_minibatches_idx(len(test[0]), valid_batch_size)

		print "%d train examples" % len(train[0])
		print "%d valid examples" % len(valid[0])
		print "%d test examples" % len(test[0])

		history_errs = []
		best_p = None
		bad_count = 0

		if validFreq == -1:
			validFreq = len(train[0]) / batch_size
		if saveFreq == -1:
			saveFreq = len(train[0]) / batch_size

		uidx = 0  # the number of update done
		estop = False  # early stop
		start_time = time.time()
		try:
			for eidx in xrange(max_epochs):
				n_samples = 0

				# Get new shuffled index for the training set.
				kf = get_minibatches_idx(len(train[0]), batch_size, shuffle=True)
	
				for _, train_index in kf:
					uidx += 1
					use_noise.set_value(1.)

					# Select the random examples for this minibatch
					y = [train[1][t] for t in train_index]
					x = [train[0][t] for t in train_index]

					# Get the data in numpy.ndarray format
					# This swap the axis!
					# Return something of shape (minibatch maxlen, n samples)
					x, mask, y = self.prepare_data(x, y)
					n_samples += x.shape[1]

					cost = f_grad_shared(x, mask, y)
					f_update(lrate)

					if numpy.isnan(cost) or numpy.isinf(cost):
						print 'NaN detected'
						return 1., 1., 1.

					if numpy.mod(uidx, dispFreq) == 0:
						print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost

					if saveto and numpy.mod(uidx, saveFreq) == 0:
						print 'Saving...',

						if best_p is not None:
							params = best_p
						else:
							params = unzip(tparams)
						numpy.savez(saveto, history_errs=history_errs, **params)
						pkl.dump(model_options, open('%s.pkl' % saveto, 'wb'), -1)
						print 'Done'

					if numpy.mod(uidx, validFreq) == 0:
						use_noise.set_value(0.)
						print 'Calculating train_err...'
						train_err = pred_error(f_pred, self.prepare_data, train, kf)
						
						print 'Calculating valid_err...'
						valid_err = pred_error(f_pred, self.prepare_data, valid,
											   kf_valid)

						print 'Calculating test_err...'
						test_err = pred_error(f_pred, self.prepare_data, test, kf_test)


						history_errs.append([valid_err, test_err])
						if (uidx == 0 or
							valid_err <= numpy.array(history_errs)[:, 0].min()):

							best_p = unzip(tparams)
							bad_counter = 0

						print ('Train ', train_err, 'Valid ', valid_err, 'Test ', test_err)

						if (len(history_errs) > patience and
							valid_err >= numpy.array(history_errs)[:-patience, 0].min()):
							bad_counter += 1
							if bad_counter > patience:
								print 'Early Stop!'
								estop = True
								break

				print 'Seen %d samples' % n_samples

				if estop:
					break

		except KeyboardInterrupt:
			print "Training interupted"

		end_time = time.time()
		if best_p is not None:
			zipp(best_p, tparams)
		else:
			best_p = unzip(tparams)

		use_noise.set_value(0.)
		kf_train_sorted = get_minibatches_idx(len(train[0]), batch_size)
		train_err = pred_error(f_pred, self.prepare_data, train, kf_train_sorted)
		valid_err = pred_error(f_pred, self.prepare_data, valid, kf_valid)
		test_err = pred_error(f_pred, self.prepare_data, test, kf_test)

		print 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err
		if saveto:
			numpy.savez(saveto, train_err=train_err,
						valid_err=valid_err, test_err=test_err,
						history_errs=history_errs, **best_p)
		print 'The code run for %d epochs, with %f sec/epochs' % (
			(eidx + 1), (end_time - start_time) / (1. * (eidx + 1)))
		print >> sys.stderr, ('Training took %.1fs' %
							  (end_time - start_time))

		return train_err, valid_err, test_err, end_time - start_time

if __name__ == '__main__':
	None
	parser = OptionParser()

	#parser.add_option("-d", "--data", action="store", type="string",
	#	dest="fname_data", default=default_fname_data)
	
	parser.add_option("-e", "--emoji", action="store", type="int",
		dest="ydim", default=10)
	parser.add_option("-s", "--n_samples", action="store", type="int",
		dest="n_samples", default=2000)
	parser.add_option("-i", "--sample_id", action="store", type="int", 
		dest="sid", default=0)

	parser.add_option("-c", "--encoder", action="store", type="string",
		dest="code", default="GBK")

	parser.add_option("-m", "--model", action="store", type="string",
		dest="fname_model", default=None)
	parser.add_option("-r", "--reload", action="store_true", 
		dest="reload_model", default=None)

	options, args = parser.parse_args()
	
	dset = dsetreader.read_dset_byarg(options.ydim, options.n_samples, options.sid)
	dataset = dsetreader.encode_dset(dset, options.code, gbkclean = True)

	coder = get_coder(options.code)
	n_words = coder.n_words

	if not options.fname_model:
		options.fname_model = '../model/model_%d_%d_%d_%s.npz'%(options.ydim, options.n_samples, options.sid, coder.name)

	lstm = ThLstm()
	train_err, valid_err, test_err, train_time = lstm.train(
			dataset = dataset,
			ydim = options.ydim,
			n_words = n_words,
			saveto = options.fname_model, 
			reload_model = options.reload_model
		)

	print '=====MARK====='
	print options.fname_model

	'''train_err = 0.11526465
	valid_err = 0.25616
	test_err = 0.5245684
	train_time = 123456483.2'''

	train_time = int(train_time)
	hr_time = train_time / 3600
	train_time %= 3600
	min_time = train_time / 60
	train_time %= 60
	sec_time = train_time

	# save dialog
	train_acc = 1. - train_err
	valid_acc = 1. - valid_err
	test_acc = 1. - test_err

	fobj = open('../result/report_%d_%d_%d_%s.txt'%(options.ydim, options.n_samples, options.sid, coder.name), 'w')

	fobj.write('Emoji\t\t%d\n'%(options.ydim))
	fobj.write('Samples\t\t%d\n'%(options.n_samples))
	fobj.write('Sid\t\t%d\n'%(options.sid))
	fobj.write('=============================\n')
	fobj.write('Train_acc\t%0.6f\n'%(train_acc))
	fobj.write('Valid_acc\t%0.6f\n'%(valid_acc))
	fobj.write('Test_acc\t%0.6f\n'%(test_acc))
	fobj.write('=============================\n')
	fobj.write('Time\t\t%d hr %d min %d sec\n'%(hr_time, min_time, sec_time))

	fobj.close()


