# -*- coding: utf-8 -*-
"""
Created on Tue May 12 10:17:32 2020

@author: marcs
"""
import GetOldTweets3 as got
import pandas as pd
import numpy as np
import datetime
import datetime as timedelta
import datetime
#from datetime import datetime
from afinn import Afinn
import os
import csv
from sentistrength import PySentiStr
import subprocess
import shlex
import time
from random import random
from twython import Twython
import time
import tweepy
import math


print(os.getcwd())
os.chdir("C:/Users/marcs/OneDrive/Bureaublad/Master/Thesis")


df = pd.DataFrame()
k=0
print("Start part 1:")
until = datetime.datetime(2019,1,1) 
since =  datetime.datetime(2018,12,31)
init_start = datetime.datetime.now()

afinn = Afinn(emoticons=True)
senti = PySentiStr()
senti.setSentiStrengthPath('C:/Users/marcs/OneDrive/Bureaublad/Master/Thesis/SentiStrength.jar') # Note: Provide absolute path instead of relative path
senti.setSentiStrengthLanguageFolderPath('C:/Users/marcs/OneDrive/Bureaublad/Master/Thesis/SentiStrength_Data/') # Note: Provide absolute path instead of relative path

for j in list(range(100000)):
    start = datetime.datetime.now()
    res = None
    while res is None:
        try:
            tweetCriteria = got.manager.TweetCriteria().setQuerySearch('$HAS')\
                                                   .setSince(since.strftime('%Y-%m-%d'))\
                                                   .setUntil(until.strftime('%Y-%m-%d'))\
                                                   .setMaxTweets(10000)\
                                                   .setEmoji("unicode")\
                                                   .setLang("en")
            tweet = got.manager.TweetManager.getTweets(tweetCriteria)
            
            time_tweet = datetime.datetime.now()
            
            print("Getting tweets", (time_tweet-start).seconds,"seconds"  )
            res = 1
        except:
            time.sleep(random() * 20)
            print("Trying again")
            pass
    
    #until = (tweet[len(tweet)-1].date)-timedelta(days = 1).strftime('%Y-%m-%d')        
    for i in list(range(len(tweet))):
        df.at[i+k,"text"] = tweet[i].text
        df.at[i+k,"id"] = tweet[i].id
        #df.at[i+k,"geo"] = tweet[i].geo
        df.at[i+k,"username"] = tweet[i].username
        df.at[i+k,"retweets"] = tweet[i].retweets
        df.at[i+k,"favorites"] = tweet[i].favorites
        df.at[i+k,"hashtags"] = tweet[i].hashtags
        df.at[i+k,"date"] = tweet[i].date.strftime('%Y-%m-%d')
        df.at[i+k,"afinn_score"] = afinn.score(tweet[i].text)
        senti_res = senti.getSentiment(tweet[i].text, score = "trinary")
        df.at[i+k,"senti_score_pos"] = senti_res[0][0]
        df.at[i+k,"senti_score_neg"] = senti_res[0][1]
        df.at[i+k,"senti_score_neu"] = senti_res[0][2]
    #df = df.drop_duplicates()    
    k=len(df)    
    
    time_proc = datetime.datetime.now()
    print("Processing tweets:", (time_proc-time_tweet).seconds," seconds" )
    
    print(since.strftime('%Y-%m-%d'), "finished")
    
    if since.year == 2015:
        time_total = (datetime.datetime.now() - init_start).seconds
        print('Finished in ', time_total, "seconds")
        break
    df.to_csv('dHASdata_2016-2018.csv')
    since = (since - datetime.timedelta(days=1))
    until =  (until - datetime.timedelta(days=1))












consumer_key = "ABWaCSqPDcxpNHSriKgvr36pv"
consumer_secret = "v7Jcti9kfkFzN1YAjFdawX4V0i671Zwq5twuNxpTGqtaFAM49T"
access_token = "1199031675763736578-ZC0G6bG4KddktrrQdFUfo2l5qU1TPy"
access_secret = "iaJBcLvbEXQADjGcg1sBLfzolCVgwg3rjRy2oSgOIT9zG"

#set up twitter api
twitter = Twython(consumer_key, consumer_secret, oauth_version=2)
ACCESS_TOKEN = twitter.obtain_access_token()
twitter = Twython(consumer_key, access_token=ACCESS_TOKEN)
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token,access_secret)

api = tweepy.API(auth)

#user related data
os.chdir("C:/Users/marcs/OneDrive/Bureaublad/Master/Thesis")
data = pd.read_csv("LMTdata2016-2018_rev3.csv") 

userlist = data['username'].unique()
iter = math.ceil(len(userlist)/100)
time_start = datetime.now()

for i in list(range(iter)):
    if i % 300==0 and i !=0:
        time_dif = int((datetime.now()-time_start).total_seconds())
        if time_dif<900:
            time.sleep(900-time_dif)
        time_start = datetime.now()            
    if i<(iter-1):
        users = twitter.lookup_user(screen_name=list(userlist[(i*100):(i*100+100)]))
    else:
        users = twitter.lookup_user(screen_name=list(userlist[(i*100):len(userlist)]))

    for user in users:
        data.loc[data['username'] == user['screen_name'],'user_id']  = user['id_str']
        data.loc[data['username'] == user['screen_name'],'user_followers'] = user['followers_count']
        data.loc[data['username'] == user['screen_name'], 'created_at'] = user['created_at']

    time_dif = int((datetime.now()-time_start).total_seconds())
    
    print(str(round(i/iter*100,2))+"%")

data.to_csv("LMTdata2016-2018_rev4.csv")

#user followers count
# for i in users:
#     try:
#         user = twitter.show_user(screen_name=i)
#         usermatrix.at[i,'created_at'] = user['created_at']
#         usermatrix.at[i, 'followers_count'] = user['followers_count']
#         usermatrix.at[i, 'friends_count'] = user['friends_count']
#         usermatrix.at[i, 'id'] = user['id']
#         usermatrix.to_csv('LMT_users_full_data.csv')
#     except:
#         pass


#dollar_users
os.chdir("C:/Users/marcs/OneDrive/Bureaublad/Master/Thesis")
data = pd.read_csv("Dollar_NVDIDA_data2016-2018_rev3.csv") 

userlist = data['username'].unique()

iter = math.ceil(len(userlist)/100)
time_start = datetime.now()
for i in list(range(iter)):
    if i % 300==0 and i !=0:
        time_dif = int((datetime.now()-time_start).total_seconds())
        if time_dif<900:
            time.sleep(900-time_dif)
        time_start = datetime.now()            
    if i<(iter-1):
        users = twitter.lookup_user(screen_name=list(userlist[(i*100):(i*100+100)]))
    else:
        users = twitter.lookup_user(screen_name=list(userlist[(i*100):len(userlist)]))

    for user in users:
        data.loc[data['username'] == user['screen_name'],'user_id']  = user['id_str']
        data.loc[data['username'] == user['screen_name'],'user_followers'] = user['followers_count']
        data.loc[data['username'] == user['screen_name'], 'created_at'] = user['created_at']
    time_dif = int((datetime.now()-time_start).total_seconds())
    
    print(str(round(i/iter*100,2))+"%")

data.to_csv("Dollar_LMT_data2016-2018_rev4.csv")




#user followers count
# for i in users:
#     try:
#         user = twitter.show_user(screen_name=i)
#         usermatrix.at[i,'created_at'] = user['created_at']
#         usermatrix.at[i, 'followers_count'] = user['followers_count']
#         usermatrix.at[i, 'friends_count'] = user['friends_count']
#         usermatrix.at[i, 'id'] = user['id']
#         usermatrix.to_csv('LMT_users_full_data.csv')
#     except:
#         pass

    

#followes of certain page
ids = []
pagenr = 0
for page in tweepy.Cursor(api.followers_ids, screen_name="@FinancialTimes").pages():
    ids.extend(page)
    pagenr = pagenr + 1
    print("Page number "+ str(pagenr) + " finished from +-" + str(1350))
    time.sleep(60)

        