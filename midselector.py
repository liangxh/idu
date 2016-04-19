#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.04.19
'''

import sys
reload(sys)
sys.setdefaultencoding('utf8')

def load_by_mid(cur, mid):
	cur.execute('SELECT text FROM microblogs WHERE mid = %s LIMIT 1'%(mid))
	res = cur.fetchone()

	return res[0]
	
def main():
	import db
	import datica
	from utils import progbar

	con = db.connect()
	cur = con.cursor()

	odname = 'data/dataset_emo/'
	

	config = datica.load_config('data/config4.txt')
	for label, eids in enumerate(config):
		for eid in eids:
			print >> sys.stderr, 'loading LABEL %d - EID %d'%(label, eid)			

			ifname = 'data/eid_mids/%d.txt'%(eid)

			ofname = odname + '%d.txt'%(eid)
			ofobj = open(ofname, 'w')

			mids = open(ifname, 'r').read().split('\n')

			pbar = progbar.start(len(mids)):
			l = 0
			for mid in mids:
				t = load_by_mid(cur, mid)
				res = blogger.extract(t)
				if res is not None:
					text, emo = res
					ofobj.write(text + '\n')
			
				l += 1
				pbar.update(l)
			pbar.finish()
			ofobj.close()

	cur.close()
	con.close()

if __name__ == '__main__':
	main()
