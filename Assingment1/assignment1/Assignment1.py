'''
Created on Feb 13, 2015

@author: santhosh
'''

import nltk
from nltk.corpus import brown
from nltk import *
import os

#Coarse Grained Tags
#Super Noun:
SNN = {"NN", "NNS", "NNP", "NNPS", "PRP", "PRP$"}
#Super Verb:
SVB = {"VB", "VBP", "VBD", "VBN", "VBZ", "VBG"}
#Super Adjective:
SJJ = {"JJ", "JJR", "JJS"}
#Super Adverb:
SRB = {"RB", "RBR", "RBS"}


number_of_train_lines = 500

#key for overall accuracy for the dictionary
OVERALL_ACCURACY = "Overall Accuracy"

MISC = '-MISC-'

#dictionary for fine_grained tags to coarse grained tags
fine_grained_to_coarse_grained_tags = {"NN":"SNN", "NNS":"SNN", "NNP":"SNN", "NNPS":"SNN", "PRP":"SNN", "PRP$":"SNN",\
                                       "VB":"SVB", "VBP":"SVB", "VBD":"SVB", "VBN":"SVB", "VBZ":"SVB", "VBG":"SVB",\
                                       "JJ":"SJJ", "JJR":"SJJ", "JJS":"SJJ",\
                                       "RB":"SRB", "RBR":"SRB", "RBS":"SRB"}


# split the annoatedWord to list of word,pos
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



#Read the data from treebank.txt and compact the data to list of list of tuples of word and pos
#if replaceByCoarseGrainedTags is True, all fine grained tags are converted to corresponding coarse grained tags
def prepareData(replaceByCoarseGrainedTags = False):
    train_data = []
    test_data = []
    diff_pos = set()
    with open(os.path.join(os.getcwd(),'data1.txt')) as f:
        lines = f.readlines()
        current_data = train_data
        current_line = 0
        for line in lines:
            annotated_words = line.split()
            words_tuples = []
            for word_idx in range(1, len(annotated_words)):
                annotatedWord = annotated_words[word_idx]
                word, posTag = split_word_and_pos(annotatedWord)
                if replaceByCoarseGrainedTags:
                    if posTag in fine_grained_to_coarse_grained_tags:
                        posTag = fine_grained_to_coarse_grained_tags.get(posTag)
                    else:
                        posTag = MISC
                diff_pos.add(posTag)
                words_tuples.append((word, posTag))
            current_data.append(words_tuples)
            current_line += 1
            if current_line == number_of_train_lines:
                current_data = test_data
    return train_data, test_data, diff_pos


# write the predictions to a file
def write_predictions_to_file(file_name, output_predictions):
    id = number_of_train_lines+1
    with open(file_name,'w') as f:
        string_out = ''
        for real,predicted in output_predictions:
            string_out = string_out+str(id)+'\t'
            for word,pos in real:
                string_out = string_out+word+'/'+pos+' '
            string_out = string_out+'\t'
            for word,pos in predicted:
                string_out = string_out+word+'/'+pos+' '
            string_out = string_out+'\n'
            id+=1
        f.write(string_out)


# create evaluation data structures
def createEvalutationStats(diff_pos, replaceByCoarseGrainedTags):
    if not replaceByCoarseGrainedTags:
        allTagsTotalDict = {key: 0 for key in diff_pos}
        correctlyPredictedTagCountDict = {key: 0 for key in diff_pos}
        confusionMatrix = {(key1, key2): 0 for key1 in diff_pos for key2 in diff_pos}
    else:
        confusionMatrix = dict()
        allTagsTotalDict = dict()
        correctlyPredictedTagCountDict = dict()
        for key1 in diff_pos:
            ckey1 = key1
            if key1 in fine_grained_to_coarse_grained_tags:
                ckey1 = fine_grained_to_coarse_grained_tags.get(key1)
            else:
                ckey1 = MISC
            allTagsTotalDict[ckey1] = 0
            correctlyPredictedTagCountDict[ckey1] = 0
            for key2 in diff_pos:
                ckey2 = key2
                if key2 in fine_grained_to_coarse_grained_tags:
                    ckey2 = fine_grained_to_coarse_grained_tags.get(key2)
                else:
                    ckey2 = MISC
                confusionMatrix[(ckey1, ckey2)] = 0
    return allTagsTotalDict, confusionMatrix, correctlyPredictedTagCountDict


#predict tags using the posTagger and populate the evaluation statistics
def predictAndEvaluate(posTagger, testData, diff_pos, replaceByCoarseGrainedTags = False):
    allTagsTotalDict, confusionMatrix, correctlyPredictedTagCountDict = \
        createEvalutationStats(diff_pos, replaceByCoarseGrainedTags)
    test_sentences_with_predictions = []
    to_be_predicted_sentences = [[word for word, pos in sentence_with_tags] for sentence_with_tags in test_data]
    predicted_tag_for_sents = posTagger.tag_sents(to_be_predicted_sentences)
    for i in range(len(predicted_tag_for_sents)):
        predicted_tag_word_pairs = predicted_tag_for_sents[i]
        sentence_with_tags = test_data[i]
        processed_predicted_sent_with_tags = []

        for j in range(len(predicted_tag_word_pairs)):
            word, predicted_tag = predicted_tag_word_pairs[j]
            word, real_tag = sentence_with_tags[j]
            if replaceByCoarseGrainedTags:
                if predicted_tag in fine_grained_to_coarse_grained_tags:
                    predicted_tag = fine_grained_to_coarse_grained_tags.get(predicted_tag)
                else:
                    predicted_tag = MISC
                if real_tag in fine_grained_to_coarse_grained_tags:
                    real_tag = fine_grained_to_coarse_grained_tags.get(real_tag)
                else:
                    real_tag = MISC

            if predicted_tag == real_tag:
                correctlyPredictedTagCountDict[predicted_tag] += 1

            confusion_matrix_entry = (real_tag, predicted_tag)
            confusionMatrix[confusion_matrix_entry] += 1
            allTagsTotalDict[real_tag] += 1
            processed_predicted_sent_with_tags.append((word,predicted_tag))

        output_for_sent = (sentence_with_tags, processed_predicted_sent_with_tags)
        test_sentences_with_predictions.append(output_for_sent)


    accuracyDict = {key: 0 for key in correctlyPredictedTagCountDict}
    for key in correctlyPredictedTagCountDict:
        if allTagsTotalDict[key] == 0:
            continue
        accuracyDict[key] = float(correctlyPredictedTagCountDict[key])/float(allTagsTotalDict[key])
    accuracyDict[OVERALL_ACCURACY] = float(sum(correctlyPredictedTagCountDict.values()))/float(sum(allTagsTotalDict.values()))

    return accuracyDict, confusionMatrix, test_sentences_with_predictions


# write evaluation tables to the CSV
def writeConfMatrixAndAccuracyTableToCSV(diff_keys, question):
    with open(os.path.join(os.getcwd(), 'accuracy_'+question+'.tsv'), 'w') as f:
        # if diff_keys != None:
        #     sorted_keys = sorted(diff_keys, key = lambda key: accuracyDict[key])
        # else:
        sorted_keys = sorted(accuracyDict.keys(), key = lambda key: accuracyDict[key])
        out1 = 'POS'+'\t'+'Accuracy'+'\n'
        for key in sorted_keys:
            out1 = out1+key+'\t'+str(accuracyDict[key])+'\n'
        f.write(out1)

    if diff_keys == None:
        diff_keys = set()
        for key in confusionMatrix.keys():
            diff_keys.add(key[0])

    with open(os.path.join(os.getcwd(), 'conf_matrix_'+question+'.tsv'), 'w') as f:
        out = '/' + '\t'
        for key1 in diff_keys:
            out = out + key1 + '\t'
        out = out + '\n'
        for key1 in diff_keys:
            out = out + key1 + '\t'
            for key2 in diff_keys:
                out = out + str(confusionMatrix[(key1, key2)]) + '\t'
            out = out + '\n'
        f.write(out)




#--------------------------------------------------------------------------------------------------------------------------
#question 1
train_data, test_data, diff_pos = prepareData()
unigramTagger = UnigramTagger(train_data, backoff=nltk.DefaultTagger('NN'))
bigramTagger = BigramTagger(train_data, backoff = unigramTagger)
accuracyDict, confusionMatrix, output_predictions = predictAndEvaluate(bigramTagger, test_data, diff_pos)
write_predictions_to_file(os.path.join(os.getcwd(),'part-I-predictions.tsv'), output_predictions)
# keys = sorted(accuracyDict.keys(),key=lambda key: accuracyDict[key],reverse=True)
# for key in keys:
#     print key,accuracyDict[key]
# print confusionMatrix
# print len(confusionMatrix.keys())
# keys = sorted(accuracyDict.keys(), key= lambda key: accuracyDict[key])
# print [(key,accuracyDict[key]) for key in keys]

# shortened keys so as to reduce the confusion matrix size
d_keys = set(['JJ', 'NN', 'NNP', 'NNPS', 'RB', 'RP', 'IN', 'VB', 'VBD', 'VBN', 'VBP'])
writeConfMatrixAndAccuracyTableToCSV(diff_keys=d_keys, question='q1')
print accuracyDict[OVERALL_ACCURACY]
print 'Question 1 done'
#question 2
#part1
train_data, test_data, diff_pos = prepareData()
unigramTagger = UnigramTagger(train_data, backoff=nltk.DefaultTagger('NN'))
bigramTagger = BigramTagger(train_data, backoff= unigramTagger)
accuracyDict, confusionMatrix, output_predictions = predictAndEvaluate(bigramTagger, test_data, diff_pos,\
                                                                      replaceByCoarseGrainedTags=True)
write_predictions_to_file(os.path.join(os.getcwd(),'Method-A-predictions.tsv'), output_predictions)
writeConfMatrixAndAccuracyTableToCSV(diff_keys=None, question='q2-p1')
# print confusionMatrix
# print len(confusionMatrix.keys())
print accuracyDict[OVERALL_ACCURACY]
print 'Question 2 Part 1 done'

#part2
train_data, test_data, diff_pos = prepareData(replaceByCoarseGrainedTags=True)
unigramTagger = UnigramTagger(train_data, backoff=nltk.DefaultTagger('SNN'))
bigramTagger = BigramTagger(train_data, backoff= unigramTagger)
accuracyDict, confusionMatrix, output_predictions = predictAndEvaluate(bigramTagger, test_data, diff_pos)
write_predictions_to_file(os.path.join(os.getcwd(),'Method-B-predictions.tsv'), output_predictions)
writeConfMatrixAndAccuracyTableToCSV(diff_keys=None, question='q2-p2')
# print confusionMatrix
# print len(confusionMatrix.keys())
print accuracyDict[OVERALL_ACCURACY]
print 'Question 2 Part 2 done'
