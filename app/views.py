# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from django.contrib.auth.decorators import login_required
from django.shortcuts import render, get_object_or_404, redirect
from django.template import loader
from django.http import HttpResponse
from django import template
from django.http import JsonResponse

from django.http import StreamingHttpResponse

import pandas as pd
import numpy as np
import json
import os


import sklearn
import pickle
from sklearn import preprocessing
import joblib

import nltk
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
nltk.download('stopwords')

from keras.models import load_model
        
from sklearn.pipeline import Pipeline, FeatureUnion
nltk.download('punkt')
# FeatureUnion combines two or more pipelines or transformers
# and is very fast!
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import StandardScaler


# Custom Transformer (Inheriting from classes)
class CleanText( BaseEstimator, TransformerMixin ):
    
  
    def __init__( self, lang = "english"):
        self.lang = lang
        self.stemmer = SnowballStemmer(self.lang)
    
    # The 'fit' method here is used to instantiate the class on the 'self' variable
    # and return the object itself
    def fit( self, X, y = None ):
        return self
    
    # Custom function: this applies the stemmer just created in the '__init__'
    # part to the 'self' variable
    def clean( self, x ):
        words   = [self.stemmer.stem(word) for word in word_tokenize(x.lower()) if word.isalpha() and word not in stopwords.words("english")]
        return " ".join(words)
    
    # Method that describes what we need this transformer to do i.e. cleaning the text
    # in the 'text' column in the data frame.
    # This will be used later on in the usage of the custom transformer
    # within the pipeline.
    def transform( self, X, y = None ):
        return X["text"].apply(self.clean)



class CustomFeatures( BaseEstimator, TransformerMixin ):
    
    # Class Constructor
    def __init__( self ):
        return
    
    # Return self nothing else to do here
    def fit( self, X, y = None ):
        return self
        
    # Method that describes what we need this transformer to do i.e.
    # returning length, digits and punctuations in the 'text' column in data frame
    def transform( self, X, y = None ):
        import pandas as pd
        f= pd.DataFrame()
        f['len']    = X['text'].str.len()
        f['digits'] = X['text'].str.findall(r'\d').str.len()
        f['punct']  = X['text'].str.findall(r'[^a-zA-Z\d\s:]').str.len()
        return f[['len','digits','punct']]


pipe_bi = Pipeline([("extract", FeatureUnion([("terms", Pipeline([('clean', CleanText()),
                                                               ('tfidf', TfidfVectorizer(ngram_range = (1,2)))])),
                                           ("custom", CustomFeatures())])),
                 ("select", SelectKBest(score_func = chi2,k = 500)),
                 ("scale", StandardScaler(with_mean = False))])



cv_RF = pickle.load(open('model_pickles/RandomForestCV.sav', 'rb'))
cv_MLP= pickle.load(open('model_pickles/MLP.sav', 'rb'))

cv_NN= load_model('model_pickles/NN_cv.sav')

cv_logistic = pickle.load(open('model_pickles/logisticCV.sav', 'rb'))
encoder = preprocessing.LabelEncoder()
encoder.classes_ = np.load('model_pickles/classes.npy', allow_pickle = True)

x_train = pd.read_pickle('model_pickles/X_train.sav')
y_train = pd.read_pickle('model_pickles/y_train.sav')

pipe_bi.fit(x_train.drop("weigth",axis=1),y_train)


print(cv_logistic)


def get_emotion(request, lyrics):

        if lyrics is not None:
            print(lyrics)
            transformed_lyrics = pipe_bi.transform(pd.DataFrame.from_dict({"text":[lyrics]}))

            y_pred_lg = cv_logistic.predict(transformed_lyrics)
            
            y_pred_rf = cv_RF.predict(transformed_lyrics)

            y_pred_mlp = cv_MLP.predict(transformed_lyrics)

           # y_pred_nn = cv_NN.predict(transformed_lyrics)
            err = "no"
            emotion_lg = str(encoder.inverse_transform(y_pred_lg)[0])
            emotion_rf = str(encoder.inverse_transform(y_pred_rf)[0])
            emotion_mlp = str(encoder.inverse_transform(y_pred_mlp)[0])
               
            y_pred_nn = cv_NN.predict(transformed_lyrics)
            emotion_nn = encoder.inverse_transform([np.argmax(y_pred_nn) for i in y_pred_nn])[0]

            print("Emotion predicted LG:",emotion_lg)
            print("Emotion predicted RF:",emotion_rf)
            print("Emotion predicted MLP:",emotion_mlp)
            
            print("Emotion predicted NN:",emotion_nn)

            context = { "errors":err,"emotion_lg": emotion_lg, 'emotion_rf':emotion_rf, 'emotion_mlp':emotion_mlp, 'emotion_nn':emotion_nn}
            return JsonResponse(context)
        else:
            return JsonResponse({'errors':"error"})


#@login_required(login_url="/login/")
def index(request):

    context = {}
    context['segment'] = 'index'

    return render(request, 'index.html',context)

#@login_required(login_url="/login/")
def pages(request):
    context = {}
    # All resource paths end in .html.
    # Pick out the html file name from the url. And load that template.
    try:

        load_template      = request.path.split('/')[-1]
        context['segment'] = load_template
        context = {}
        context['segment'] = 'data'




        html_template = loader.get_template( load_template )
        return HttpResponse(html_template.render(context, request))

    except template.TemplateDoesNotExist:

        html_template = loader.get_template( 'page-404.html' )
        return HttpResponse(html_template.render(context, request))

    except:

        html_template = loader.get_template( 'page-500.html' )
        return HttpResponse(html_template.render(context, request))
