# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 15:10:54 2020

@author: marcs
"""
import emoji
import os
import pandas as pd

os.chdir("C:/Users/marcs/OneDrive/Bureaublad/Master/Thesis")
name = "dHAS"
data = pd.read_csv(name+ "data2016-2018_rev.csv")
#data = pd.read_csv("LMTdata2016-2018_rev.csv") 

def get_emoji_score(s,emoji_lexicon):
    emojistring = ''.join(c for c in s if(c in emoji.UNICODE_EMOJI))
    emojiscore = 0
    if emojistring != '':
        emojilist = emojistring.split()
        for i in emojilist:
            emojiscore = emojiscore + ((emoji_lexicon[emoji_lexicon['Emoji'] == i]['Positive']-emoji_lexicon.loc[emoji_lexicon['Emoji'] == i]['Negative'])/emoji_lexicon.loc[emoji_lexicon['Emoji'] == i]['Occurrences']).sum()
    return emojiscore

emoji_scores = [0] * len(data.index)
os.chdir("C:/Users/marcs/OneDrive/Bureaublad/Master/Thesis/EmojiSentiment")
emoji_lexicon = pd.read_csv("Emoji_Sentiment_Data_v1.0.csv") 

for i in data.index:
    emoji_scores[i] = get_emoji_score(data['text'][i],emoji_lexicon)
    
os.chdir("C:/Users/marcs/OneDrive/Bureaublad/Master/Thesis")
data['emoji'] = emoji_scores

data.to_csv(name+"data2016-2018_rev2.csv")
