#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.02.02
Modified: 2016.02.02
Description: easy access to get progressbar 
'''

from progressbar import *

widgets = ['Progress: ', Percentage(), ' ', Bar(marker='>'), ' ', ETA()]

def start(maxval):
	pbar = ProgressBar(widgets = widgets, maxval = maxval).start()
	return pbar


if __name__ == '__main__':
	loops = 100000
	interval = 100
	maxval = loops / interval

	pbar = start(maxval)
	
	for i in range(loops):
		if i % interval == 0:
			pbar.update(i / interval)

	pbar.update(maxval)
	pbar.finish()
