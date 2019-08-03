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
import re,time

now = str(datetime.datetime.now())

df = pd.read_csv('mergeData.csv', delimiter=',', names=['Data', 'Label']);
# print(df)

first_col = df.ix[1:, 0]
# print(first_row)
second_col = df.ix[1:, 1]
second_col = second_col.fillna(0)
print(second_col.isna().sum())
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
    

def split_doc():
    for data in first_col:
        return_string = data_preprocessing(data)
        each_docs = return_string.split()
        string_after_remove_word=stop_word_remove(each_docs)
        # print(string_after_remove_word)
        data_with_split.append(string_after_remove_word)
    return data_with_split  # it returns arr of each docs with spleted words


word_lists = []
word_lists = split_doc()
length_of_docs = len(word_lists)


#####################################
# print(word_lists)

def individual_words():
    my_set = set.union(*map(set, word_lists))  # seperate each individual words from data to make matrix
    return my_set


def set_to_list():
    my_set_list = individual_words()
    convert_into_list = list(my_set_list)
    return convert_into_list


individual_word_list = set_to_list()


def count_occurence_of_word_vocab():
    my_set = individual_words()
    doc = {}
    word_dict = {}
    for i in range(len(word_lists)):
        for word in word_lists[i]:
            word_dict = dict.fromkeys(my_set, 0)

    for count_word_value in word_lists:
        for word in count_word_value:
            if word in word_dict:
                word_dict[word] += 1
    return word_dict


word_dict = count_occurence_of_word_vocab()



length_word_dict = len(word_dict)

def vectorizer_docs(line):
    vectorizer_docs = []
    matrix_doc = []
    for word in individual_word_list:
        if word in line:
            vectorizer_docs.append(1)
        else:
            vectorizer_docs.append(0)
    return vectorizer_docs
    vectorizer_docs.clear()


doc_vec1 = []
doc_vec2 = []
for line in word_lists:
    doc_vec1 = vectorizer_docs(line)
    doc_vec2.append(doc_vec1)
# print(doc_vec2)

dict1={}
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
for each_line in word_lists:
    tf = computeTf(each_line)
    tf_vec += tf
    #######################################3
# print("Term Frequency")
# print(tf_vec)

countIdfforwordvalue = {}
word_dict = count_occurence_of_word_vocab()
my_set = individual_words()


def computeCountDict(word_dict, word_lists):
    countIdfforword = {}
    for i in range(1, len(my_set)):
        countIdfforword = dict.fromkeys(my_set, 0)
    for word, value in word_dict.items():
        for each_line_item in word_lists:
            if word in each_line_item:
                countIdfforword[word] += 1
        # else:
        #   countIdfforword[word] = 1
    return countIdfforword


countIdfforwordvalue = computeCountDict(word_dict, word_lists)

#  #  return no of doc conatin word for each word
#  def doc_contain_word(parameter_word):
#       word_value_in_each_doc = countIdfforwordvalue.get(parameter_word)
#       return word_value_in_each_doc


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
for each_line in word_lists:
    idf = computeIdf(each_line)
    idf_vec += idf
################################################
# print("idf vector")
# print(idf_vec)

TfIdf_vec = []
def computeTfIdf(Tfvec, Idfvec):
    TfIdf_vec = [a * b for a, b in zip(Tfvec, Idfvec)]
    return TfIdf_vec


tfidf_vector_for_each_docs = []
tfidf_vector_collection = []
for tf_list, idf_list in zip(tf_vec, idf_vec):  # zip helps to iteration two different collection samultaneously
    tfidf_vector_for_each_docs = computeTfIdf(tf_list, idf_list)
    tfidf_vector_collection.append(tfidf_vector_for_each_docs)
    
def feature_labels():
    features = np.array(tfidf_vector_collection)
    labels_string = np.array(second_col)
    labels_list = [int(int_labels) for int_labels in labels_string]
    labels = np.array(labels_list)
    array_length = len(features)
    return features,labels,array_length
