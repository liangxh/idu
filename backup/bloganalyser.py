#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.03.28
'''

import sys
reload(sys)
sys.setdefaultencoding('utf8')
import json
import numpy as np

import blogger
from utils import progbar

def load_blogs():
	def load_blog_lines():
		lines = []
		for i in range(3):
			fname = 'data/blogs/blogs_subset_%d.txt'%(i)
			lines.extend(open(fname, 'r').readlines())
		return lines

	lines = load_blog_lines()
	blogs = []
	
	pbar = progbar.start(len(lines))
	for i, line in enumerate(lines):
		blogs.append(json.loads(line))
		pbar.update(i + 1)
	pbar.finish()

	return blogs

def main():
	blogs = load_blogs()

	blog_emos = {} # 
	comm_emos = {} # tf of emo in comments
	
	yes_blog = 0
	yes_comm = []
	yes_emo_comm = []

	no_blog = 0
	no_comm = []
	no_emo_comm = []

	def inc(d, k):
		if d.has_key(k):
			d[k] += 1
		else:
			d[k] = 1

	def incs(d, ks):
		for k in ks:
			inc(d, k)

	pbar = progbar.start(len(blogs))
	for i, blog in enumerate(blogs):
		n_emo_comm = 0 # number of comments with emoticon

		# comments
		for comm in blog['comments']:
			content = comm['text']
	
			res = blogger.simple_extract(content)
			if res is not None:
				text, es = res
				
				incs(comm_emos, es)
				n_emo_comm += 1
		
		# text
		res = blogger.simple_extract(blog['text'])
		if res is not None:
			yes_blog += 1
			yes_comm.append(blog['comments_count'])	
			yes_emo_comm.append(n_emo_comm)		

			text, es = res
			incs(blog_emos, es)
		else:
			no_blog += 1
			no_comm.append(blog['comments_count'])
			no_emo_comm.append(n_emo_comm)

		pbar.update(i + 1)
	pbar.finish()

	report = ''
	report += 'number of blogs with emoticons: %d (%.2f%%)\n'%(yes_blog, 100. * yes_blog / (yes_blog + no_blog))
	report += 'number of blogs without emoticons: %d (%.2f%%)\n'%(no_blog, 100. * no_blog / (yes_blog + no_blog))
	report += 'average number of comments of blogs with emoticons: %.2f\n'%(np.mean(yes_comm))
	report += 'average number of comments of blogs without emoticons: %.2f\n'%(np.mean(no_comm))
	report += 'average number of comments with emoticons of blogs with emoticons: %.2f\n'%(np.mean(yes_emo_comm))
	report += 'average number of comments with emoticons of blogs without emoticons: %.2f\n'%(np.mean(no_emo_comm))

	print report

	def export_emo(emos, fname):
		n_emos = sum(emos.values())
		cum = 0		

		fobj = open(fname, 'w')

		emof = sorted(emos.items(), key = lambda k: -k[1])
		for i, item in enumerate(emof):
			emo, f = item
			cum += f
			fobj.write('%d. %s (%d %.2f%% %.2f%%)\n'%(i + 1, emo, f, 100. * f / n_emos, 100. * cum / n_emos))

		fobj.close()

	export_emo(blog_emos, 'output/bloganalyse_blogemos.txt')
	export_emo(comm_emos, 'output/bloganalyse_commemos.txt')

if __name__ == '__main__':
	main()

