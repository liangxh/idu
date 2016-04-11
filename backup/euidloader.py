#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.02.23
'''

import re
import sys
reload(sys)
sys.setdefaultencoding('utf8')

from const import DIR_MUID

def load_euids(eids):
	'''
	load {eid: {mid: uid, mid: uid, ...}, ...} from dataset/muid/EID.txt
	'''
	euids = {}
	for eid in eids:
		fname = DIR_MUID + '%d.txt'%(eid)
		lines = open(fname, 'r').read().split('\n')

		uids = {}
		for line in lines:
			params = line.split(' ')
			mid = params[0]
			uid = params[1]
			uids[mid] = uid

		euids[eid] = uids

	return euids
		

if __name__ == '__main__':
	pass
