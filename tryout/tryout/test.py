'''
Created on Feb 13, 2015

@author: santhosh
'''


import nltk
from nltk.corpus import brown
from nltk import *
nltk.data.path.append("/media/santhosh/Data/workspace/nltk_data")

text = word_tokenize("And now for something completely different")
print nltk.pos_tag(text)