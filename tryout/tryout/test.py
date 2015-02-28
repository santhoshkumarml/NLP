'''
Created on Feb 13, 2015

@author: santhosh
'''


import nltk
from nltk.corpus import brown
from nltk import *
import sys
nltk.data.path.append("/media/santhosh/Data/workspace/nltk_data")

def split_word_and_pos(annotatedWord):
    prevChar = ''
    wordAndPos = []
    currentBuf = ''
    for ch in annotatedWord:
        if ch == '/' and prevChar != '\\':
            wordAndPos.append(currentBuf)
            currentBuf = ''
        else:
            currentBuf = currentBuf + ch
        prevChar = ch
        
    wordAndPos.append(currentBuf)
    
    return wordAndPos
            
with open('../data1.txt') as f:
    lines = f.readlines()
    train_data = []
    test_data = []
    current_data = train_data
    number_of_train_lines = 500
    current_line = 0
    for line in lines:
        annotated_words = line.split()
        words_tuples = []
        for word_idx in range(1, len(annotated_words)):
            annotatedWord = annotated_words[word_idx]
            word,pos = split_word_and_pos(annotatedWord)
            words_tuples.append((word,pos))
    
        current_data.append(words_tuples)
        current_line += 1
        if current_line == number_of_train_lines:
            current_data = test_data

# tagged_sents = nltk.corpus.treebank.tagged_sents()
# train_data = tagged_sents[:500]
# print train_data
bigramTagger = BigramTagger(train_data)
tagAccuracyDict = dict()
allTagsTotalDict = dict()
confusionMatrix = dict()
for sent_with_tags in test_data:
    sent = [word for word, pos in sent_with_tags]
    predicted_tag_pairs = bigramTagger.tag(sent)
    for i in range(len(sent_with_tags)):
        
        predicted_tag = predicted_tag_pairs[i][1]
        real_tag = sent_with_tags[i][1]
        
        if predicted_tag == real_tag:
            if predicted_tag not in tagAccuracyDict:
                tagAccuracyDict[predicted_tag] = 0
            tagAccuracyDict[predicted_tag] += 1
            
        confusion_matrix_entry1 = (real_tag, predicted_tag)
        confusion_matrix_entry2 = (predicted_tag, real_tag)
        
        if confusion_matrix_entry1 not in confusionMatrix:
            confusionMatrix[confusion_matrix_entry1] = 0
            confusionMatrix[confusion_matrix_entry2] = 0
        confusionMatrix[confusion_matrix_entry1] += 1
        
        if real_tag not in allTagsTotalDict:
            allTagsTotalDict[real_tag] = 0
        allTagsTotalDict[real_tag] += 1
        
    
print tagAccuracyDict
print allTagsTotalDict
print confusionMatrix
        
    
#print bigramTagger.tag(treebank_sents[2007])