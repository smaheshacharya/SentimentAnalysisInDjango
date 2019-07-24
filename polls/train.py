import pandas as pd
import math as mth
from functools import reduce
from collections import Counter
import numpy as np
from sklearn.naive_bayes import GaussianNB
import pickle
import datetime
import random
import sys
from sklearn import metrics
import re

global labels
now = str(datetime.datetime.now())

df = pd.read_csv('merge.csv', delimiter=',', names=['Data', 'Label']);
# print(df)
first_col = df.ix[1:, 0]
# print(first_row)
second_col = df.ix[1:, 1]
# print(second_row)
data_with_split = []
each_docs = []
stop_words_split_final = []


#data cleaning method
def data_preprocessing(string):
    text = re.sub('\,|\@|\-|\"|\'| \)|\(|\)| \{| \}| \[| \]|!|‘|’|“|”| \:-|\?|।|/|\—', '', string)
    return text
#hello
def stop_word_remove(array_element):
    stop_words = set(['मैले','छ','र','तर','को','मा','म','त','यो','ती','न','पनि','छन्','अब','के','छु','भए','यस','ले','लागि','भन','हरे','हरेक','हो','तथा','भएको','गरेको','भने','गर्न','गर्ने','यी','का','गरि','कि','जुन','गरेर','छैन','अलग','आए','अझै','गए','गरौं','गर्छ','गर्छु','कतै','जब','जबकि','जसको','तल','भर','जे','जो','ठीक','धेरै','नजिकै','नत्र'])
    array_element_set = set(array_element)
    final_list = list(array_element_set.difference(stop_words))
    return final_list
    

def SPLITDOCUMENT():
    for data in first_col:
        return_string = data_preprocessing(data)
        each_docs = return_string.split()
        string_after_remove_word=stop_word_remove(each_docs)
        # print(string_after_remove_word)
        data_with_split.append(string_after_remove_word)
    return data_with_split  # it returns arr of each docs with spleted words


word_arrays = []
word_arrays = SPLITDOCUMENT()
length_of_docs = len(word_arrays)


def individual_words():
    my_set = set.union(*map(set, word_arrays))  # seperate each individual words from data to make matrix
    return my_set


def set_to_list():
    my_set_list = individual_words()
    convert_into_list = list(my_set_list)
    return convert_into_list


individual_word_array = set_to_list()


def count_occurence_of_word_vocab():
    my_set = individual_words()
    doc = {}
    word_dict = {}
    for i in range(len(word_arrays)):
        for word in word_arrays[i]:
            word_dict = dict.fromkeys(my_set, 0)

    for count_word_value in word_arrays:
        for word in count_word_value:
            if word in word_dict:
                word_dict[word] += 1
    return word_dict


word_dict = count_occurence_of_word_vocab()
# print(len(word_dict))


length_word_dict = len(word_dict)

def vectorizer_docs(line):
    vectorizer_docs = []
    matrix_doc = []
    for word in individual_word_array:
        if word in line:
            vectorizer_docs.append(1)
        else:
            vectorizer_docs.append(0)
    return vectorizer_docs
    vectorizer_docs.clear()


doc_vec1 = []
doc_vec2 = []
for line in word_arrays:
    doc_vec1 = vectorizer_docs(line)
    doc_vec2.append(doc_vec1)
# print(doc_vec2)


def computeTf(docs_list):
    tf_vec = []
    tf_each_doc_vec = []
    doc_word_count = len(docs_list)
    count_each_word = Counter(docs_list)  # counter count the word in list how many times it occure
    for each_word,val in word_dict.items():
        if each_word in docs_list:
            count = count_each_word.get(each_word)
            tf_vec.append(count / float(doc_word_count))
        else:
            tf_vec.append(0)
    tf_each_doc_vec.append(tf_vec)
    return tf_each_doc_vec


tf = []
tf_vec = []
for each_line in word_arrays:
    tf = computeTf(each_line)
    tf_vec += tf
# print("Term Frequency")
# print(tf_vec)

countIdfforwordvalue = {}
word_dict = count_occurence_of_word_vocab()
my_set = individual_words()


def computeCountDict(word_dict, word_arrays):
    countIdfforword = {}
    for i in range(1, len(my_set)):
        countIdfforword = dict.fromkeys(my_set, 0)
    for word, value in word_dict.items():
        for each_line_item in word_arrays:
            if word in each_line_item:
                countIdfforword[word] += 1
        # else:
        # 	countIdfforword[word] = 1
    return countIdfforword


countIdfforwordvalue = computeCountDict(word_dict, word_arrays)


#  #  return no of doc conatin word for each word
#  def doc_contain_word(parameter_word):
# 		word_value_in_each_doc = countIdfforwordvalue.get(parameter_word)
# 		return word_value_in_each_doc


def computeIdf(docs_list):
    idf_vec = []
    idf_each_doc_vec = []
    for each_word,val in word_dict.items():
        if each_word in docs_list:
            word_value_in_each_doc = countIdfforwordvalue.get(each_word)
            idf_vec.append(mth.log(length_of_docs / word_value_in_each_doc))
        else:
            idf_vec.append(0)
    idf_each_doc_vec.append(idf_vec)
    return idf_each_doc_vec


idf = []
idf_vec = []
for each_line in word_arrays:
    idf = computeIdf(each_line)
    idf_vec += idf
# print("Inverse document frequency")
# print(len(idf_vec[0]))
# print(len(idf_vec[2]))
# print(len(idf_vec[3]))
# print(len(idf_vec[4]))
# print(len(idf_vec[5]))
# print(len(idf_vec[6]))

TfIdf_vec = []


def TFIDF(Tfvec, Idfvec):
    TfIdf_vec = [a * b for a, b in zip(Tfvec, Idfvec)]
    return TfIdf_vec

tfidf_vector_for_each_docs = []
tfidf_vector_collection = []
for tf_list, idf_list in zip(tf_vec, idf_vec):  # zip helps to iteration two different collection samultaneously
    tfidf_vector_for_each_docs = TFIDF(tf_list, idf_list)
    tfidf_vector_collection.append(tfidf_vector_for_each_docs)
# make model with sk-learn
def feature_labels():
    features = np.array(tfidf_vector_collection)
    labels_string = np.array(second_col)
    labels_list = [int(int_labels) for int_labels in labels_string]
    labels = np.array(labels_list)
    array_length = len(features)
    return features,labels,array_length
# print(type(features))

# features_taken_len = int(array_length * 80 / 100)  # 80% of data make for train 20% remening data for testing
# feature_array_train = features[:features_taken_len]  # 80% of data make for train 20% remening data for testing
# labels_array_train = labels[:features_taken_len]
# feature_array_test = features[features_taken_len:]  # 80% of data make for train 20% remening data for testing
# labels_array_test =  labels[features_taken_len:]

# print(feature_array_train.shape)
# print(labels_array_train.shape)

# final_labels = labels_array_test.reshape(1,-1)
# final_feature_test = feature_array_test.reshape(1,-1)
# print(type(feature_array_test))
# print(type(labels_array_test))
#
# print(feature_array_test)
# print("train")
# print(feature_array_train)
# print(labels_array_train)
# print("test")
# print(feature_array_test)
# print(labels_array_test)


# Naive byes classifier sklearn
#train model
