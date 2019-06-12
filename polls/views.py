from django.contrib.auth import get_user_model
from django.shortcuts import render, redirect
from django.http import JsonResponse
from rest_framework.views import APIView
from rest_framework.response import Response
import requests
from .train import *
import codecs, json
from numpyencoder import NumpyEncoder


# import matplotlib.pyplot as plt
with open('classify_data.pickle', 'rb') as pickle_saved_data:
    unpickled_data = pickle.load(pickle_saved_data)
def index(request):
    return render(request,'index.html')

def testfunction(request ,*arg,**kwarg):
    if request.method == "GET":
        data_from = request.GET["test_data"]
        test_percent = int(data_from)
        features, labels, array_length = feature_labels()
        features_taken_len = int(array_length * test_percent / 100)  # 80% of data make for train 20% remening data for testing
        feature_array_train = features[:features_taken_len]  # 80% of data make for train 20% remening data for testing
        labels_array_train = labels[:features_taken_len]
        feature_array_test = features[features_taken_len:]  # 80% of data make for train 20% remening data for testing
        labels_array_test =  labels[features_taken_len:]
        naive_byes = GaussianNB()  # create  object  from  GaussianNb  class
        TrainData = naive_byes.fit(feature_array_train, labels_array_train)

        classifier_data = open("classify_data.pickle", "wb")
        pickle.dump(TrainData, classifier_data)
        classifier_data.close()
        naive_byes_test = GaussianNB()
        TestData = naive_byes_test.partial_fit(feature_array_test, labels_array_test, classes=np.unique(labels_array_test))
        predict_result = TrainData.predict(feature_array_test)
        dict_for_idf = {}
        def count_each_word_each_doc():
            i = 1
            for each_line_for_idf in word_lists:
                dict_for_idf[i] = {}
                count_each_word_for_idf = Counter(each_line_for_idf)
                for each_word_of_line_for_idf in each_line_for_idf:
                    count_for_idf = count_each_word_for_idf.get(each_word_of_line_for_idf)
                    dict_for_idf[i][each_word_of_line_for_idf] = count_for_idf
                i = i+1
            return dict_for_idf
        dict_for_idf_final = count_each_word_each_doc()

        naive_byes_test = GaussianNB()
        TestData = naive_byes_test.partial_fit(feature_array_test, labels_array_test, classes=np.unique(labels_array_test))


            #predict data using pickle file
        predict_result = unpickled_data.predict(feature_array_test)

        #precision
        precision = metrics.precision_score(predict_result,labels_array_test ,average='weighted')
        #recall

        recall = metrics.recall_score(predict_result,labels_array_test,average='weighted')


        #f score

        f_score = 2*(precision*recall)/(precision+recall)

        labels = [  'Precision', 'Recall', 'F-Score']
        default_items = [precision,recall,f_score]
        data = {

        "labels":labels,
        "default":default_items,

        }
        return JsonResponse(data)

def input_tf(input_data):

    each_input_word = []
# change into array of word
    each_input_word = input_data.split()

#input data from user
    length_input_data = len(each_input_word)

    count_each_inputword = Counter(each_input_word)
    input_data_tfvec = []
# tf_each_input_word = []
#TF computation of input data

    for word,val in word_dict.items():#where word_dict is all the word collection from data set
        if word in each_input_word:
            count = count_each_inputword.get(word)
            input_data_tfvec.append(count / float(length_input_data))
        else:
            input_data_tfvec.append(0)
    return input_data_tfvec




def input_idf(input_data):
    idf_vec_input_data = []
    idf_each_doc_vec_input_data = []
    for each_word_input_data,val in word_dict.items():
        if each_word_input_data in input_data:
            word_value_in_each_doc_input_data = countIdfforwordvalue.get(each_word_input_data)
            idf_each_doc_vec_input_data.append(mth.log(length_of_docs / word_value_in_each_doc_input_data))
        else:
            idf_each_doc_vec_input_data.append(0)
    return idf_each_doc_vec_input_data





def predict(request,*arg,**kargs):
    if request.method == "GET":
        search_word = request.GET["data_from_form"]

        tf_value_of_input_data = input_tf(search_word)
        idf_value_of_input_data = input_idf(search_word)

        tfidf_input_vec = [a * b for a, b in zip(tf_value_of_input_data, idf_value_of_input_data)]

        value_for_predict = np.array(tfidf_input_vec).reshape(1,-1)
        predict = unpickled_data.predict(value_for_predict)

        int_data = int(np.asarray(predict))

        result = {
        "data":int_data,

        }
        return JsonResponse(result)
