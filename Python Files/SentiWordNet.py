# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 12:23:48 2020

@author: marcs
"""
from nltk.corpus import stopwords   
import nltk
import os
import pandas as pd
from nltk.tag import pos_tag
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
import datetime as datetime



os.chdir("C:/Users/marcs/OneDrive/Bureaublad/Master/Thesis")
name = "dHas"
data = pd.read_csv(name+"data2016-2018_rev2.csv") 

stop = stopwords.words('english')   # initialize english stopwords

def penn_to_wn(tag):
    """
    Convert between the PennTreebank tags to simple Wordnet tags
    """
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None

SWN_score = [0]*len(data.index)
SWN_score_bin = ['l']*len(data.index)
init_start = datetime.datetime.now()
for i in data.index:
    wordlist = str(data['text'][i])
    tokens = nltk.tokenize.word_tokenize(wordlist)
    tags = pos_tag(tokens)
    totalScore = 0
    count_words_included = 0
    
    for word in tags:
        tag = penn_to_wn(word[1])
        synset_forms = list(swn.senti_synsets(word[0], tag))
    
        if synset_forms == []:
    
            continue
    
        synset = synset_forms[0] 
        totalScore = totalScore + synset.pos_score() - synset.neg_score()
    
        count_words_included = count_words_included +1
    
    final_dec = ''
    
    if count_words_included == 0:
    
        final_dec = 'N/A'
    
    elif totalScore == 0:
    
        final_dec = 'Neu'        
    
    elif totalScore/count_words_included < 0:
    
        final_dec = 'Neg'
    
    elif totalScore/count_words_included > 0:
    
        final_dec = 'Pos'
    SWN_score[i] = totalScore
    SWN_score_bin[i] = final_dec
    seconds = (datetime.datetime.now() - init_start).seconds
    if i % 1000 == 0:
        print(str(round((i/len(data.index))*100,2))+"% in " + str(seconds) + " seconds")

data['SWN'] = SWN_score



os.chdir("C:/Users/marcs/OneDrive/Bureaublad/Master/Thesis")
data['SWN'] = SWN_score

data.to_csv(name+'data2016-2018_rev3.csv')
