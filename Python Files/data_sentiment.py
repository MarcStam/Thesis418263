# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 11:25:20 2020

@author: marcs
"""

import os
import pandas as pd
import numpy as np
import datetime
from datetime import datetime
import datetime as timedelta
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import classification_report
import math
import matplotlib.pyplot as plt


#print(os.getcwd())
os.chdir("C:/Users/marcs/OneDrive/Bureaublad/Master/Thesis")

data_full_LMT = pd.read_csv("LMTdata2016-2018_rev5.csv") 
dol_data_LMT = pd.read_csv("Dollar_LMT_data2016-2018_rev5.csv")

data_full_NVDA = pd.read_csv("NVIDIAdata2016-2018_rev5_2.csv") 
dol_data_NVDA = pd.read_csv("dNVIDIAdata2016-2018_rev5_2.csv")

data_full_HAS = pd.read_csv("HASdata2016-2018_rev5_2.csv") 
dol_data_HAS = pd.read_csv("dHASdata2016-2018_rev5.csv")

datasets = [data_full_NVDA,dol_data_NVDA,data_full_HAS,dol_data_HAS,data_full_LMT,dol_data_LMT]
datasets_names = ['data_full_NVDA','dol_data_NVDA','data_full_HAS','dol_data_HAS','data_full_LMT','dol_data_LMT']
scores = ['afinn_score','senti_score_neu','OpFi_score','emoji','SWN']

res = pd.DataFrame()
j=0
for data in datasets: 
    for score in scores: 
        res.at[score+" pos",datasets_names[j]] = str(round(sum(data[score]>0) /len(data.index),3)*100) + "%"
        res.at[score+" neu",datasets_names[j]] = str(round(sum(data[score]==0) /len(data.index),3)*100) + "%"
        res.at[score+" neg",datasets_names[j]] = str(round(sum(data[score]<0) /len(data.index),3)*100) + "%"
            
    j = j + 1 

res.to_csv("data_sentiment.csv")
