#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import theano

def main():
	a = np.zeros((104912, 128)).astype(theano.config.floatX)
	ta = theano.shared(a, borrow=True)
	
	return ta

if __name__ == '__main__':
	main()
