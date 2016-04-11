#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.04.11
'''

import sys
reload(sys)
sys.setdefaultencoding('utf8')
import json
import commands

import blogger
from utils import progbar


def extract(ifname, ofname):
	n_lines = int(commands.getoutput('grep -cF "" %s'%(ifname)))
	if n_lines == 0:
		return

	pbar = progbar.start(n_lines)
	l = 0

	datalist = []

	ofobj = open(ofname, 'w')
	with open(ifname, 'r') as ifobj:
		for line in ifobj:
			blog = json.loads(line)
			n_text = len(blog['text'])

			for i in range(1, n_text):
				res = blogger.extract(blog['text'][i])
				if res is not None:
					datum = {}
					datum['text'] = res[0]
					datum['emo'] = res[1]
					
					above = []			
					re_id = i
					while blog['re'][re_id] is not None:
						re_id = blog['re'][re_id]
						above.append(blog['text'][re_id])
					datum['above'] = above

					follow = []
					last_id = i					
					for j in range(i + 1, n_text):
						if blog['re'][j] == last_id:
							follow.append(blog['text'][j])
							last_id = j

					datum['follow'] = follow

					ofobj.write(json.dumps(datum) + '\n')
	
					#for k in range(n_text):
					#	print '%d. (-> %s) %s'%(k, blog['re'][k], blog['text'][k])
					#
					#print 'above:'
					#print '\n'.join(above)
					#print 'follow:'
					#print '\n'.join(follow)
					#print 

				#if i > 100:
				#	return

			l += 1
			pbar.update(l)
		pbar.finish()
	
	ofobj.close()

def main():
	#ifname = 'data/blogs/0s.txt'
	#ofname = 'data/blogs/out0s.txt'
	
	ifname = sys.argv[1]
	ofname = sys.argv[2]	

	extract(ifname, ofname)

if __name__ == '__main__':
	main()

