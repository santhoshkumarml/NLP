import StringIO
__author__ = 'santhosh'

import nltk
from nltk import *
import re
import csv
from cStringIO import StringIO

nltk.data.path.append('/media/santhosh/Data/workspace/nltk_data')

NOVEL_BASE = '/media/santhosh/Data/workspace/nlp_project/novels'
NOVEL_META = 'novel_meta.txt'
dataset_pattern = r'[*]+DATASET:.*[*]+'
folder_pattern = r'[*]+.*[*]+'
entry_pattern = r'(SUCCESS|FAILURE): FileName:.*,Title:.*,Author:.*,Language:.*,DownloadCount:.*'
SUCCESS_PATTERN = 'SUCCESS'
FAILURE_PATTERN = 'FAILURE'

KEY_TOKENS = 'FileName|Title|Author|Language|DownloadCount'
LANG_TOKEN = 'Language'


def fixMetaInfoRecord(tokens):
    wrong_split_idx = set()
    for i in range(len(tokens)):
        if ':' not in tokens[i]:
            wrong_split_idx.add(i)
    
    mergeable_idx = dict()
    for idx in wrong_split_idx:
        i = 1
        while idx - i in wrong_split_idx:
            i += 1
        
        if idx - i not in mergeable_idx:
            mergeable_idx[idx - i] = []
        mergeable_idx[idx - i].append(idx)
    
    for key in mergeable_idx:
        for idx in mergeable_idx[key]:
            tokens[i] = tokens[i] +','+tokens[idx]
            
    removing_idxs = sorted(list(wrong_split_idx), reverse = True)
    for idx in removing_idxs:
        del tokens[idx]


def processMetaInfoRecord(meta_dict_for_dataset, classification, line):
    line = line.replace(classification+':','')
    tokens =  line.split(',')
    
    fixMetaInfoRecord(tokens)
      
    meta_dict_for_file = None
    for token in tokens:
        key,value = token.split(':',1)
        key,value = key.strip(), value.strip()
        if key == 'FileName':
            meta_dict_for_dataset[value] = dict()
            meta_dict_for_file = meta_dict_for_dataset[value]
        else:
            meta_dict_for_file[key] = value
    meta_dict_for_file['class'] = classification

def loadInfoFromMetaFile():
    meta_dict = dict()
    with open(os.path.join(NOVEL_BASE,NOVEL_META)) as f:
        lines = f.readlines()
        dataset = None
        for line in lines:
            if re.match(dataset_pattern, line):
                line_strip = line.replace('*','')
                dataset = line_strip.split(':')[1].strip()
                meta_dict[dataset] = {SUCCESS_PATTERN:dict(), FAILURE_PATTERN:dict()}
            elif re.match(entry_pattern,line):
                if SUCCESS_PATTERN+':' in line:
                    processMetaInfoRecord(meta_dict[dataset], SUCCESS_PATTERN, line)
                elif FAILURE_PATTERN+':' in line:
                    processMetaInfoRecord(meta_dict[dataset], FAILURE_PATTERN, line)
    return meta_dict

def listGenreWiseFileNames():
    genre_folders = [f for f in os.listdir(NOVEL_BASE) if not os.path.isfile(os.path.join(NOVEL_BASE,f))]
    genre_to_file_list = dict()
    for genre_folder in genre_folders:
        fullPath_to_genre_folder = os.path.join(NOVEL_BASE,genre_folder)
        multiFolderLevels = [os.path.join(fullPath_to_genre_folder,f)\
                              for f in os.listdir(fullPath_to_genre_folder)\
                              if not os.path.isfile(os.path.join(fullPath_to_genre_folder,f))]
        success_failure_folders = [os.path.join(multiFolderLevel,f)\
                                    for multiFolderLevel in multiFolderLevels\
                                    for f in os.listdir(multiFolderLevel)\
                                    if not os.path.isfile(os.path.join(multiFolderLevel,f))]
        onlyFiles = [(os.path.join(success_failure_folder,f),f)\
                                    for success_failure_folder in success_failure_folders\
                                    for f in os.listdir(success_failure_folder)\
                                    if os.path.isfile(os.path.join(success_failure_folder,f))]
        genre_file_dict_key = genre_folder.replace("_",' ')
        genre_to_file_list[genre_file_dict_key] = onlyFiles
        
    return genre_to_file_list

def readGenreBasedFiles(genre_to_file_list, meta_dict):
    for genre in genre_to_file_list:
        print 'Genre:',genre
        meta_dict_for_genre = meta_dict[genre]
        print '--------------------------------------------------------------'
        for genre_file_path,genre_file_name in genre_to_file_list[genre]:
            print 'File:',genre_file_path
            if genre_file_name not in meta_dict_for_genre or meta_dict_for_genre[genre_file_name][LANG_TOKEN] != 'en':
                continue
            with open(genre_file_path) as f:
                filelines = f.readlines()
                for fileline in filelines:
                    tokens = nltk.word_tokenize(fileline)
                    print nltk.pos_tag(tokens)
            print '--------------------------------------------------------------'
        print '--------------------------------------------------------------'
        print '--------------------------------------------------------------'

meta_dict = loadInfoFromMetaFile()
genre_to_file_list = listGenreWiseFileNames()
readGenreBasedFiles(genre_to_file_list, meta_dict)