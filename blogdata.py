#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Description: a script to turn blogs downloaded from weiboparser into {MID, TEXT, RE}
Created: 2016.04.07
'''

import sys
reload(sys)
sys.setdefaultencoding('utf8')

import re
import json
import commands

'''
class BlogData:
	def __init__(self, uid, mid, ):
		self.uid = 
		self.mid = mid
		self.content = []

	def 
'''

def same_name(name1, name2):
	if name1 == name2:
		return True
	elif name1 is None or name2 is None:
		return False

	for i in range(len(name1) - 1):
		if name1[i:i+2] in name2:
			 return True
	return False

def print_raw(j):
	invalidname = set()

	for k, v in j.items():
		if k == 'comments':
			print 'comments:'
			for i, com in enumerate(v):
				print '\t%d. %s to %s: %s'%(i + 1, com['from_name'], com['to_name'], com['text'])

				for name in [com['from_name'], com['to_name']]:
					if name is not None and not j['ids'].has_key(name):
						invalidname.add(name)
		elif k == 'ids':
			print 'ids: '
			for ik, iv in v.items():
				print '\t %s: %s'%(ik, iv)
		else:
			print k, ': ', v
			if k == 'uid':
				for ki, vi in j['ids'].items():
					if v == vi:
						print 'uname: %s'%(ki)
						break

	print 'number of changed names: %d'%(len(invalidname))

def clean_comments(comments):
	n_comments = len(comments)
	batch = 20
	st = 0
	
	comms = []
	while st <  n_comments:
		ed = min(st + batch, n_comments)
		cs = comments[st:st + batch]
		cs.reverse()
		comms.extend(cs)
		st = ed
	
	comms.reverse()

	pattern_reply = re.compile(u'^回(复|覆) ?@([^\:]+):')
	for i, comment in enumerate(comms):
		if comment['to_name'] is None:
			m = pattern_reply.match(comment['text'])
			if m:
				comment['to_name'] = m.group(2)
				comment['text'] = pattern_reply.sub('', pattern_reply.sub('', comment['text']))

	return comms

def load_from_rawblogs(j):
	uid = j['uid']
	text = j['text']
	ids = j['ids']
	comments = j['comments']

	# match the username of uid
	uname = ''
	for ki, vi in j['ids'].items():
		if uid == vi:
			uname = ki
			break

	# matching the old name and the new
	undefined_names = set(j['ids'].keys())
	undefined_tonames = set()
	for comm in comments:
		if comm['to_name'] is not None:
			if j['ids'].has_key(comm['to_name']):
				undefined_names.discard(comm['to_name']) 
			else:
				undefined_tonames.add(comm['to_name'])

	#print '############ TEST ################'
	if len(undefined_tonames) == 0 or len(undefined_names) == 0:
		#print 'CLEAR'
		pass
	else:
		#print ' / '.join(undefined_tonames)
		#print ' / '.join(undefined_names)

		to_newname = {}

		for oldname in undefined_tonames:
			newname = None
			for name in undefined_names:
				if same_name(name, oldname):
					undefined_names.discard(name)
					to_newname[oldname] = name
					break
		undefined_tonames -= set(to_newname.keys())

		if len(undefined_tonames) == 1 and len(undefined_names) == 1:
			to_newname[list(undefined_tonames)[0]] = list(undefined_names)[0]

		for comm in comments:
			if to_newname.has_key(comm['to_name']):
				comm['to_name'] = to_newname[comm['to_name']]

		#for k, v in to_newname.items():
		#	print k, '-->', v


	content = []
	reply_to = []

	content.append(text)
	reply_to.append(None)

	to_ed = set()
	n_comments = len(comments)
	for i, comment in enumerate(comments):
		to_name = comment['to_name']
		from_name = comment['from_name']
		text = comment['text']

		j = i - 1
		flag_uname = same_name(from_name, uname)
		flag_toNone = to_name is None
		flag_toUndef = to_name in undefined_tonames

		content.append(text)
		while (j >= 0):
			if j in to_ed:
				j -= 1
				continue

			# this is rough but quick, still xD
			last_comm = comments[j]
			f1 = (flag_uname and last_comm['to_name'] is None and (flag_toNone or flag_toUndef))
			f2 = (flag_toNone and last_comm['to_name'] == from_name)

			f4 = (flag_toUndef and last_comm['from_name'] in undefined_names)

			f5 = ((last_comm['from_name'], last_comm['to_name']) == (from_name, to_name))
			f6 = ((last_comm['from_name'], last_comm['to_name']) == (to_name, from_name))
			f7 = (last_comm['to_name'] is None and last_comm['from_name'] == to_name)
			f8 = (last_comm['to_name'] is None and last_comm['from_name'] == from_name and uname == to_name)

			#if False: #(i, j) == (1, 0):
			#	print i, j, ':',  f1, f2, f4, f5, f6, f7, f8
			#	print j, last_comm['from_name'], last_comm['to_name'], last_comm['text']
			#	print i, from_name, to_name, text

			if (f1 or f2 or f4 or f5 or f6 or f7 or f8):
				if f4:
					oname = comment['to_name']
					nname = last_comm['from_name']
					for k in range(j, n_comments):
						if comments[k]['to_name'] == oname:
							comments[k]['to_name'] = nname
					
					comment['to_name'] = last_comm['from_name']
					undefined_names.discard(comment['to_name'])
				
				reply_to.append(j + 1)
				to_ed.add(j)
				break
			j -= 1

		if j < 0:
			reply_to.append(0)

	
	'''l = 0
	for c, r in zip(content, reply_to):
		if l == 0:
			print '%d. %s (-> %s)'%(l, c, r)
		elif comments[l - 1]['to_name'] == None:
			print '%d. %s: %s (-> %s)'%(l, comments[l - 1]['from_name'], c, r)
		else:		
			print '%d. %s re-%s: %s (-> %s)'%(l, comments[l - 1]['from_name'], comments[l - 1]['to_name'], c, r)
		l += 1'''

	return content, reply_to

def test():
	jid = int(sys.argv[1])

	lines = open('data/blog/bcomm_sample.txt', 'r').readlines()
	js = [json.loads(line) for line in lines]

	j = js[jid]
	j['comments'] = clean_comments(j['comments'])

	print_raw(j)
	print '###############################################'
	
	content, reply_to = load_from_rawblogs(j)

def preprocess(ifname, ofname):
	from utils import progbar
	n_lines = int(commands.getoutput('grep -c \'\\n\' %s'%(ifname)))
	
	pbar = progbar.start(n_lines)
	l = 0

	ofobj = open(ofname, 'w')
	ifobj = open(ifname, 'r')
	for line in ifobj:
		j = json.loads(line)
		j['comments'] = clean_comments(j['comments'])
		content, reply_to = load_from_rawblogs(j)
		
		outdata = {'mid':j['mid'], 'text':content, 're':reply_to}
		ofobj.write(json.dumps(outdata) + '\n')

		l += 1
		pbar.update(l)
	pbar.finish()

	ifobj.close()
	ofobj.close()

def main():
	ifname = sys.argv[1]
	ofname = sys.argv[2]

	preprocess(ifname, ofname)
		
if __name__ == '__main__':
	main()
