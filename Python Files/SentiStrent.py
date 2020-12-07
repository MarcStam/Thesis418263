# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 14:29:57 2020

@author: marcs
"""
import pandas as pd
from sentistrength import PySentiStr
import os
import numpy as np


os.chdir("C:/Users/marcs/OneDrive/Bureaublad/Master/Thesis")

data_full = pd.read_csv("HASdata2016-2018_rev5.csv") 

senti = PySentiStr()
senti.setSentiStrengthPath('C:/Users/marcs/OneDrive/Bureaublad/Master/Thesis/SentiStrength.jar') # Note: Provide absolute path instead of relative path
senti.setSentiStrengthLanguageFolderPath('C:/Users/marcs/OneDrive/Bureaublad/Master/Thesis/SentiStrength_Data/') # Note: Provide absolute path instead of relative path


for i in data_full.index:
    if np.isnan(data_full.at[i,'senti_score_neu']):
        senti_res = senti.getSentiment(data_full.text[i], score = "trinary")
        data_full.at[i,"senti_score_pos"] = senti_res[0][0]
        data_full.at[i,"senti_score_neg"] = senti_res[0][1]
        data_full.at[i,"senti_score_neu"] = senti_res[0][2]
    if i % 100 ==0:
        print(round(i/len(data_full.index)*100,2))
        
data_full.to_csv("HASdata2016-2018_rev5_2.csv")