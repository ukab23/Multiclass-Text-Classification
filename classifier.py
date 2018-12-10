import re
import os
import sys
import glob
import json
import string
import numpy as np
import nltk
import pickle
import dill
import sklearn
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from many_stop_words import get_stop_words
from nltk.tokenize import wordpunct_tokenize
from pprint import pprint
from sklearn.externals import joblib


class Classifier():
    def __init__(self):
        # self.labels = list(filter(lambda x: not (x.endswith("~") or x.startswith("client")), os.listdir("./classifier/command_classifier/data/")))
        self.labels = [  
                         "date.txt",
                         "day.txt",
                         "time.txt",
                         
        ]
        # print(self.labels)
        self.vectorizer = TfidfVectorizer( max_features = None, strip_accents = 'unicode',
                            analyzer = "word", ngram_range=(1,1), use_idf = 1, smooth_idf = 1, stop_words='english')
        self.model_folder = "./classifier/command_classifier/models"
        self.model_path = self.model_folder+"/command_classifier.joblib"
        

        xdata = []
        ydata = []
        train_path ="./data/"
        for file in self.labels:
        	# print("File Exists")
        	with open(train_path+file,"r") as f:
        		data = f.readlines()
        		data = [i.replace("\n","") for i in data]
        		data = [i for i in data if i != ""]
        		for i in data:
        			xdata.append(i)
        			ydata.append(self.labels.index(file))
        vectors_train = self.vectorizer.fit_transform(xdata)
        self.classifier = OneVsRestClassifier( LogisticRegression( C = 10.0, multi_class = "multinomial", solver = "lbfgs" ) )
        self.classifier.fit(vectors_train, ydata)
        # print("logostic classifier accuracy=====>",self.classifier.score(vectors_train, ydata))
    def test(self, query):
        classifier_result = {}
        vectors_test = self.vectorizer.transform([query])
        pred = self.classifier.predict(vectors_test)
        predp = self.classifier.predict_proba(vectors_test)
        predp = predp.tolist()[0]
        max_ind =  predp.index(max(predp))
        classifier_result["class"] = self.labels[max_ind][:-4]
        classifier_result["prediction"] = {self.labels[max_ind]: round(max(predp)*100.0, 2)*0.6}
        # with_confidence,with_confusion = max(predp),max(filter(lambda n:n != max(predp),predp))
        # confident_on,confused_on = modifier_dict[self.labels[predp.index(with_confidence)]],modifier_dict[self.labels[predp.index(with_confusion)]]
        # classifier_result =  [confident_on, confused_on] if (with_confidence - with_confusion) < .05 else [confident_on]
        # print dict(zip([confident_on, confused_on], [with_confidence,with_confusion])) , classifier_result
        print (classifier_result)
        return classifier_result

if __name__ == '__main__':
        while(True):
            result = Classifier()
            qr = input("=>")
            result.test(qr)