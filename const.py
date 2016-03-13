#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Xihao Liang
Created: 2016.01.20
Modified: 2016.01.28
Description: stores const variables for sharing
'''

############## FILENAME or DIRECTORY #########################

DIR_DATA = 'data/'

DIR_EID_MIDS = DIR_DATA + 'eid_mids/'    # mids of the top 200 emoticons are stored in EID.txt 
DIR_EID_UIDS = DIR_DATA + 'eid_uids/'    # uids of the blogs in eid_mids are stored as uids (e.g. uids[mid] = uid) in EID.txt
DIR_COMMENTS = DIR_DATA + 'comments/'    # 

PKL_EMO_MIDS = DIR_DATA + 'emo_mids.pkl' # store all {EMO: [MIDs, ], }
TXT_EID = DIR_DATA + 'eid.txt'           # phrases of emoticons are listed in the order of EID

URL_EMO = 'http://weibo.com/aj/mblog/face?type=face&_wv=5&ajwvr=6&__rnd=1453464129146'
JSON_EMO = DIR_DATA + 'emoticon.json'    # raw data parsed from URL_EMO

DIR_OUTPUT = 'output/'

DIR_DATASET = DIR_DATA + 'dataset/'
DIR_TEXT = DIR_DATASET + 'text/'
DIR_MUID = DIR_DATASET + 'muid/'
DIR_TOKEN = DIR_DATASET + 'token/'

PKL_TFCODER = DIR_DATASET + 'tfcoder.pkl'
PKL_TFDATA = DIR_DATASET + 'tfdata.pkl'

################ CONST VARIABLE ##############################

N_EMO = 90

#WEIBO_COOKIE = open(DIR_DATA + 'weibo_cookie.txt', 'r').read()
#WEIBO_MYID = '1784127181'

STATUS_COUNT_MAX = 1000
STATUS_COUNT_2ND = 1098

TOTAL_USERS = 1666637
TOTAL_BLOGS = 156365006

################# ABONDENED ##################################

PKL_REPORT = DIR_DATA + 'report.pkl'
TXT_STATUS_COUNT = DIR_DATA + 'status_count.txt'  # UID STATUS_COUNT 

