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

def get_data(request ,*arg,**kwargs):
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


def predict(request,*arg,**kargs):
    if request.method == "GET":
        search_word = request.GET["data_from_form"]
        print(search_word)
        each_input_word = []
# change into array of word
        each_input_word = search_word.split()

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
# to make predict input value similar as our training sample we use reshape
        value_for_predict = np.array(input_data_tfvec).reshape(1,-1)
        predict_data = unpickled_data.predict(value_for_predict)
        int_data = int(np.asarray(predict_data))
        result = {
        "data":int_data,

        }
        return JsonResponse(result)
