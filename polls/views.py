from django.contrib.auth import get_user_model
from django.shortcuts import render, redirect
from django.http import JsonResponse
from rest_framework.views import APIView
from rest_framework.response import Response
import requests
from .train import *

# import matplotlib.pyplot as plt

def index(request):
    return render(request,'index.html')
    
def get_data(request ,*arg,**kwargs):
    naive_byes_test = GaussianNB()
    TestData = naive_byes_test.partial_fit(feature_array_test, labels_array_test, classes=np.unique(labels_array_test))

    with open('classify_data.pickle', 'rb') as pickle_saved_data:
        unpickled_data = pickle.load(pickle_saved_data)
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
