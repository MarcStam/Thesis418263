# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 16:12:05 2020

@author: 418263ms
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

os.chdir("C:/Users/marcs/OneDrive/Bureaublad/Master/Thesis")

#data
data_full = pd.read_csv("NVIDIAdata2016-2018_rev5.csv") 
dol_data = pd.read_csv("dNVIDIAdata2016-2018_rev5.csv")
FT_users =  pd.read_csv("ids_Financial_times.csv") 
FT_user_ids = FT_users['ids']       
# users = pd.read_csv('LMT_users_full_data.csv')
# dollar_users =  pd.read_csv('LMT_users_full_data.csv')

#for Nvida full set
for i in data_full.index:
    try:
        data_full.at[i,'date'] = datetime(int(data_full['date'][i][6:10]),int(data_full['date'][i][0:2]),int(data_full['date'][i][3:5])).strftime('%Y-%m-%d')
    except:
        try:
            data_full.at[i,'date'] = datetime(int(data_full['date'][i][5:9]),int(data_full['date'][i][0:1]),int(data_full['date'][i][2:4])).strftime('%Y-%m-%d')
        except:
            try:
                data_full.at[i,'date'] = datetime(int(data_full['date'][i][5:9]),int(data_full['date'][i][0:2]),int(data_full['date'][i][3:4])).strftime('%Y-%m-%d')
            except:
                try:
                    data_full.at[i,'date'] = datetime(int(data_full['date'][i][4:8]),int(data_full['date'][i][0:1]),int(data_full['date'][i][2:3])).strftime('%Y-%m-%d')
                except:
                    pass;


data_full['weight'] = 0.0
data_full['zero_fol'] = 0
for i in data_full.index:
    if i % 1000 == 0:
        print(i/len(data_full.index))
    try:
        days_active = int((datetime(2020,8,25) - datetime.strptime(data_full['created_at'][i][4:10]+", "+data_full['created_at'][i][26:30],'%b %d, %Y')).days)
    except:
        pass;
    try:
        data_full.at[i,'weight'] = int(data_full.at[i,'user_followers']) *  ( datetime(int(data_full['date'][i][0:4]),int(data_full['date'][i][5:7]),int(data_full['date'][i][8:10])) -datetime.strptime(data_full['created_at'][i][4:10]+", "+data_full['created_at'][i][26:30],'%b %d, %Y')).days/days_active
    except:
        try:
             data_full.at[i,'weight'] = int(data_full.at[i,'user_followers']) *  ( datetime(int(data_full['date'][i][6:10]),int(data_full['date'][i][0:2]),int(data_full['date'][i][3:5])) -datetime.strptime(data_full['created_at'][i][4:10]+", "+data_full['created_at'][i][26:30],'%b %d, %Y')).days/days_active
        except:
            try:
                data_full.at[i,'weight'] = int(data_full.at[i,'user_followers']) *  ( datetime(int(data_full['date'][i][4:8]),int(data_full['date'][i][0:1]),int(data_full['date'][i][2:3])) -datetime.strptime(data_full['created_at'][i][4:10]+", "+data_full['created_at'][i][26:30],'%b %d, %Y')).days/days_active
            except:
                try:
                    data_full.at[i,'weight'] = int(data_full.at[i,'user_followers']) *  ( datetime(int(data_full['date'][i][5:9]),int(data_full['date'][i][0:1]),int(data_full['date'][i][2:4])) -datetime.strptime(data_full['created_at'][i][4:10]+", "+data_full['created_at'][i][26:30],'%b %d, %Y')).days/days_active
                except:
                    try:
                        data_full.at[i,'weight'] = int(data_full.at[i,'user_followers']) *  ( datetime(int(data_full['date'][i][5:9]),int(data_full['date'][i][0:2]),int(data_full['date'][i][3:4])) -datetime.strptime(data_full['created_at'][i][4:10]+", "+data_full['created_at'][i][26:30],'%b %d, %Y')).days/days_active
                    except:
                        pass;
    try:
        if (datetime(int(data_full['date'][i][6:10]),int(data_full['date'][i][0:2]),int(data_full['date'][i][3:5])) == datetime.strptime(data_full['created_at'][i][4:10]+", "+data_full['created_at'][i][26:30],'%b %d, %Y')):
            data_full.at[i,'zero_fol'] = 1
    except:
        pass;
data_full['weight'] = data_full['weight']/max(data_full['weight'])    

data_full[(data_full['weight'] ==0) & (data_full['zero_fol']!=0)] = data_full[(data_full['weight'] ==0) & (data_full['zero_fol']==0)]['weight'].mean()

for i in data_full.index: 
    if len(data_full.at[i,'date']) < 5:
        data_full = data_full.drop(i)
    if i % 1000 == 0:
        print(round(i/len(data_full.index),2))

data_full.to_csv("NVIDIAdata2016-2018_rev5_2.csv")

# length2 = len(data_full[data_full['weight'] ==0].index)
# index = data_full[data_full['weight'] ==0].index
# mean = data_full[data_full['weight'] !=0]['weight'].mean()
# for i in  index:
#     data_full.at[i,'weight'] = mean
#     if i % 100:
#         print(round(i/length2,2))
    
     

#dollardata weights
dol_data['weight'] = 0
dol_data['zero_fol'] = 0
for i in dol_data.index:
    if i % 500 == 0:
        print(i/len(dol_data.index))
    try:
        days_active = int((datetime(2020,8,25) - datetime.strptime(dol_data['created_at'][i][4:10]+", "+dol_data['created_at'][i][26:30],'%b %d, %Y')).days)
    except:
        pass;
    try:
        dol_data.at[i,'weight'] = int(dol_data.at[i,'user_followers']) *  ( datetime(int(dol_data['date'][i][0:4]),int(dol_data['date'][i][5:7]),int(dol_data['date'][i][8:10])) -datetime.strptime(dol_data['created_at'][i][4:10]+", "+dol_data['created_at'][i][26:30],'%b %d, %Y')).days/days_active
    except:
        try:
             data_full.at[i,'weight'] = int(dol_data.at[i,'user_followers']) *  ( datetime(int(dol_data['date'][i][6:10]),int(dol_data['date'][i][0:2]),int(dol_data['date'][i][3:5])) -datetime.strptime(dol_data['created_at'][i][4:10]+", "+dol_data['created_at'][i][26:30],'%b %d, %Y')).days/days_active
        except:
            pass;
    try:
        if (datetime(int(dol_data['date'][i][6:10]),int(dol_data['date'][i][0:2]),int(dol_data['date'][i][3:5])) == datetime.strptime(dol_data['created_at'][i][4:10]+", "+dol_data['created_at'][i][26:30],'%b %d, %Y')):
            dol_data.at[i,'zero_fol'] = 1
    except:
        pass;
dol_data['weight'] = dol_data['weight']/max(dol_data['weight']) 
dol_data[(dol_data['weight'] ==0) & (dol_data['zero_fol']!=0)] = dol_data[(dol_data['weight'] ==0) & (dol_data['zero_fol']==0)]['weight'].mean()      
dol_data.to_csv("dNVIDIAdata2016-2018_rev5_2.csv")

data_full_FT = pd.DataFrame()
data_full['FT'] = 0
time_start = datetime.now()
for i in data_full.index:
    if np.any(FT_user_ids == data_full['user_id'][i]):
        data_full['FT'][i] = 1
    if i % 1000 == 0:
        seconds = str(round((datetime.now() - time_start).total_seconds(),0))
        print(str(i/len(data_full.index)*100)+" % in " + seconds +" seconds with length:" + str(sum(data_full['FT'])) )
data_full_FT = data_full[data_full['FT'] == 1]
data_full_FT.to_csv("NVIDIA_FT_Data2016-2018")
data_full_no_rt = data_full.sort_values(by=['date']).drop_duplicates(subset=['text']).sort_values(by=['Unnamed: 0'])
