#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.03.19
Description: keep login to the tunet at interval
'''

import time
import thulogin 
from optparse import OptionParser

def main():
	optparser = OptionParser()
	optparser.add_option('-u', '--user', action='store', type='string', dest='user')
	optparser.add_option('-p', '--password', action='store', type='string', dest='password')
	optparser.add_option('-i', '--interval', action='store', type='int', dest='interval', default = 15 * 60)

	opts, args = optparser.parse_args()
	
	while True:
		try:
			thulogin.login(opts.user, opts.password)
			time.time(opts.interval)
		except KeyboardInterrupt:
			thulogin.logout()

if __name__ == '__main__':
	main()

