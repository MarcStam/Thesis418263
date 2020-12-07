# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 14:07:20 2020

@author: marcs
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 20:32:17 2020

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
#os.chdir("./Downloads")
dataset = ''
#data

#LMT
data_full = pd.read_csv("LMTdata2016-2018_rev5.csv") 
dol_data = pd.read_csv("Dollar_LMT_data2016-2018_rev5.csv")
data_full['senti_score'] = data_full['senti_score_pos'] + data_full['senti_score_neg']
dol_data['senti_score'] = dol_data['senti_score_pos'] + dol_data['senti_score_neg']
ret_SP = pd.read_csv("^GSPC.csv") 
ret_LMT = pd.read_csv("LMT.csv")
data_full_FT = pd.read_csv("LMT_FT_Data2016-2018.csv")
data_full_no_rt = data_full.sort_values(by=['date']).drop_duplicates(subset=['text']).sort_values(by=['Unnamed: 0'])
dol_data_no_rt = dol_data.sort_values(by=['date']).drop_duplicates(subset=['text']).sort_values(by=['Unnamed: 0'])

#NVIDIA
data_full = pd.read_csv("NVIDIAdata2016-2018_rev5_2.csv") 
dol_data = pd.read_csv("dNVIDIAdata2016-2018_rev5_2.csv")
data_full['senti_score'] = data_full['senti_score_pos'] + data_full['senti_score_neg']
dol_data['senti_score'] = dol_data['senti_score_pos'] + dol_data['senti_score_neg']
ret_SP = pd.read_csv("^GSPC.csv") 
ret_LMT = pd.read_csv("NVDA.csv")
data_full_FT = pd.read_csv("NVIDIA_FT_Data2016-2018.csv")
data_full_no_rt = data_full.sort_values(by=['date']).drop_duplicates(subset=['text']).sort_values(by=['Unnamed: 0'])
dol_data_no_rt = dol_data.sort_values(by=['date']).drop_duplicates(subset=['text']).sort_values(by=['Unnamed: 0'])

#HAS
data_full = pd.read_csv("HASdata2016-2018_rev5_2.csv") 
dol_data = pd.read_csv("dHASdata2016-2018_rev5.csv")
data_full['senti_score'] = data_full['senti_score_pos'] + data_full['senti_score_neg']
dol_data['senti_score'] = dol_data['senti_score_pos'] + dol_data['senti_score_neg']
ret_SP = pd.read_csv("^GSPC.csv") 
ret_LMT = pd.read_csv("HAS.csv")
data_full_FT = pd.read_csv("HAS_FT_Data2016-2018.csv")
data_full_no_rt = data_full.sort_values(by=['date']).drop_duplicates(subset=['text']).sort_values(by=['Unnamed: 0'])
dol_data_no_rt = dol_data.sort_values(by=['date']).drop_duplicates(subset=['text']).sort_values(by=['Unnamed: 0'])


# for i in users.index:
#     users.at[i,'days_active'] = int((datetime(2020,8,8) - datetime.strptime(users['created_at'][i][4:10]+", "+users['created_at'][i][26:30],'%b %d, %Y')).days)
    
# for i in data_full.index:
#     if i % 1000 == 0:
#         print(i/len(data_full.index))
#     try:
#         user_data = users[users['Unnamed: 0'] == data_full['username'][i]]
#         data_full['weight'] = int(user_data['user_followers']) *  ( datetime(int(data_full['date'][i][0:4]),int(data_full['date'][i][5:7]),int(data_full['date'][i][8:10])) -datetime.strptime(user_data['created_at'][i][4:10]+", "+ user_data['created_at'][i][26:30],'%b %d, %Y')).days/user_data['days_active']
#     except:
#         continue;

        
#initializer
companies = ['NVIDIA','LMT','HAS']
aggregates = ['TSV','neg_aggr','neg_aggr2','sumaggr']
scores = ['afinn_score','senti_score','OpFi_score','emoji','SWN']
output_set = ['intra','abnormal', 'multi','multi2']
datasets = ['data_full','dol_data','data_full_FT','dol_data_no_rt','data_full_no_rt']
weight_opt = ['weighted','non-weighted']
lags = ['lags3','lags4','lags5','lags6']
model = ""
length = len(companies)*len(aggregates)*len(scores)*len(output_set)*len(datasets)*len(weight_opt)*len(lags)

def divide_checker(y):
    if y == 0:
        return 1e50
    else:
        return y

#
#afinn = Afinn(emoticons=True)
#senti = PySentiStr()
#senti.setSentiStrengthPath('C:/Users/marcs/OneDrive/Bureaublad/Master/Thesis/SentiStrength.jar') # Note: Provide absolute path instead of relative path
#senti.setSentiStrengthLanguageFolderPath('C:/Users/marcs/OneDrive/Bureaublad/Master/Thesis/SentiStrength_Data/') # Note: Provide absolute path instead of relative path
#


#train and test cutoff
cutoff = datetime(2017,1,1).strftime('%Y-%m-%d')
cutoff2 = datetime(2017,7,1).strftime('%Y-%m-%d')
cutoff3 = datetime(2018,1,1).strftime('%Y-%m-%d')
cutoff4 = datetime(2019,1,1).strftime('%Y-%m-%d')
# test_data = data.loc[data['date']>= cutoff]
# train_data = data.loc[data['date']< cutoff]






def trading_sim(y_pred, trading_df, alpha = 0):
    trading_df.index = trading_df['date']
    if type(y_pred) != np.ndarray and type(y_pred) != np.array and isinstance(y_pred, list) == False:
        y_pred.index = trading_df['date']
    trading_df['y_pred'] = y_pred
    ROR  = sum(trading_df[trading_df['y_pred'] == 1.0]['1_day_dif']) - sum(trading_df[trading_df['y_pred'] == 0.0]['1_day_dif'])
    profit = sum(trading_df[trading_df['y_pred'] == 1.0]['abs_dif_1_day']) - sum(trading_df[trading_df['y_pred'] == 0.0]['abs_dif_1_day'])
    trading_df['inv_rev'] = [0]*len(trading_df.index)
    trading_df['inv_rev'][trading_df['y_pred'] == 1.0] = trading_df[trading_df['y_pred'] == 1.0]['1_day_dif']
    trading_df['inv_rev'][trading_df['y_pred'] == 0.0] = - trading_df[y_pred == 0.0]['1_day_dif']
    rev_sum = trading_df.groupby(['month']).sum()['inv_rev']
    riskfree = 1.01**(1/12)-1
    sharpe = (rev_sum.mean()-riskfree)/np.std(rev_sum-riskfree)
    return [ROR, sharpe, profit]

def trading_sim_multi(y_pred, trading_df, alpha = 0):
    trading_df.index = trading_df['date']
    if type(y_pred) != np.ndarray and type(y_pred) != np.array and isinstance(y_pred, list) == False:
        y_pred.index = trading_df['date']
    trading_df['y_pred'] = y_pred
    ROR  = sum(trading_df[trading_df['y_pred'] == 5]['1_day_dif']) + sum(trading_df[trading_df['y_pred'] == 4]['1_day_dif'])- sum(trading_df[trading_df['y_pred'] == 1]['1_day_dif']) - sum(trading_df[trading_df['y_pred'] == 2]['1_day_dif'])
    profit = 5*sum(trading_df[trading_df['y_pred'] == 5]['abs_dif_1_day']) + sum(trading_df[trading_df['y_pred'] == 4]['abs_dif_1_day'])- 5*sum(trading_df[trading_df['y_pred'] == 1]['abs_dif_1_day']) - sum(trading_df[trading_df['y_pred'] == 2]['abs_dif_1_day'])
    trading_df['inv_rev'] = [0]*len(trading_df.index)
    trading_df['inv_rev'][trading_df['y_pred'] == 5] =  5*trading_df[trading_df['y_pred'] == 5]['1_day_dif'] 
    trading_df['inv_rev'][trading_df['y_pred'] == 4] = trading_df[trading_df['y_pred'] == 4]['1_day_dif']
    trading_df['inv_rev'][trading_df['y_pred'] == 1] = - 5* trading_df[y_pred == 1]['1_day_dif']
    trading_df['inv_rev'][trading_df['y_pred'] == 2] = - trading_df[y_pred == 2]['1_day_dif']
    rev_sum = trading_df.groupby(['month']).sum()['inv_rev']
    riskfree = 1.01**(1/12)-1
    sharpe = (rev_sum.mean()-riskfree)/np.std(rev_sum-riskfree)
    return [ROR, sharpe, profit]

def trading_sim_multi2(y_pred, trading_df, alpha = 0):
    trading_df.index = trading_df['date']
    if type(y_pred) != np.ndarray and type(y_pred) != np.array and isinstance(y_pred, list) == False:
        y_pred.index = trading_df['date']
    trading_df['y_pred'] = y_pred
    ROR  = sum(trading_df[trading_df['y_pred'] == 3]['1_day_dif']) - sum(trading_df[trading_df['y_pred'] == 1]['1_day_dif'])
    profit = sum(trading_df[trading_df['y_pred'] == 3]['abs_dif_1_day']) - sum(trading_df[trading_df['y_pred'] == 1]['abs_dif_1_day'])
    trading_df['inv_rev'] = [0]*len(trading_df.index)
    trading_df['inv_rev'][trading_df['y_pred'] == 3] = trading_df[trading_df['y_pred'] == 3]['1_day_dif']
    trading_df['inv_rev'][trading_df['y_pred'] == 1] = - trading_df[y_pred == 1]['1_day_dif']
    rev_sum = trading_df.groupby(['month']).sum()['inv_rev']
    riskfree = 1.01**(1/12)-1
    sharpe = (rev_sum.mean()-riskfree)/np.std(rev_sum-riskfree)
    return [ROR, sharpe, profit]




# vectorizer = TfidfVectorizer()
# response =  vectorizer.fit_transform

# l = train_data.groupby(['date']).mean()
# test_data.groupby(['date']).mean()

# vectorizer = TfidfVectorizer()
# X = vectorizer.fit_transform(train_data['text'])
# afinn.score(X['0'])




#----------------------------------------------- SECTION 1 ----------------------------------------------------------------------- 
# Determining the best 10 models per company over the first 1.5 years of data.
res_section1 = pd.DataFrame()
nr = 1
dates = dol_data['date'].unique()
for company in companies:
    result_matrix = pd.DataFrame()
    #LMT
    if company == 'LMT':
        data_full = pd.read_csv("LMTdata2016-2018_rev5.csv") 
        dol_data = pd.read_csv("Dollar_LMT_data2016-2018_rev5.csv")
        data_full['senti_score'] = data_full['senti_score_pos'] + data_full['senti_score_neg']
        dol_data['senti_score'] = dol_data['senti_score_pos'] + dol_data['senti_score_neg']
        ret_SP = pd.read_csv("^GSPC.csv") 
        ret_LMT = pd.read_csv("LMT.csv")
        data_full_FT = pd.read_csv("LMT_FT_Data2016-2018.csv")
        data_full_FT['senti_score'] = dol_data['senti_score_pos'] + dol_data['senti_score_neg']
        data_full_no_rt = data_full.sort_values(by=['date']).drop_duplicates(subset=['text']).sort_values(by=['Unnamed: 0'])
        dol_data_no_rt = dol_data.sort_values(by=['date']).drop_duplicates(subset=['text']).sort_values(by=['Unnamed: 0'])
        
    #NVIDIA
    if company == 'NVIDIA':
        data_full = pd.read_csv("NVIDIAdata2016-2018_rev5_2.csv") 
        dol_data = pd.read_csv("dNVIDIAdata2016-2018_rev5_2.csv")
        data_full['senti_score'] = data_full['senti_score_pos'] + data_full['senti_score_neg']
        dol_data['senti_score'] = dol_data['senti_score_pos'] + dol_data['senti_score_neg']
        ret_SP = pd.read_csv("^GSPC.csv") 
        ret_LMT = pd.read_csv("NVDA.csv")
        data_full_FT = pd.read_csv("NVIDIA_FT_Data2016-2018.csv")
        data_full_FT['senti_score'] = dol_data['senti_score_pos'] + dol_data['senti_score_neg']
        data_full_no_rt = data_full.sort_values(by=['date']).drop_duplicates(subset=['text']).sort_values(by=['Unnamed: 0'])
        dol_data_no_rt = dol_data.sort_values(by=['date']).drop_duplicates(subset=['text']).sort_values(by=['Unnamed: 0'])
    
    #HAS
    if company == 'HAS':
        data_full = pd.read_csv("HASdata2016-2018_rev5_2.csv") 
        dol_data = pd.read_csv("dHASdata2016-2018_rev5.csv")
        data_full['senti_score'] = data_full['senti_score_pos'] + data_full['senti_score_neg']
        dol_data['senti_score'] = dol_data['senti_score_pos'] + dol_data['senti_score_neg']
        ret_SP = pd.read_csv("^GSPC.csv") 
        ret_LMT = pd.read_csv("HAS.csv")
        data_full_FT = pd.read_csv("HAS_FT_Data2016-2018.csv")
        data_full_FT['senti_score'] = dol_data['senti_score_pos'] + dol_data['senti_score_neg']
        data_full_no_rt = data_full.sort_values(by=['date']).drop_duplicates(subset=['text']).sort_values(by=['Unnamed: 0'])
        dol_data_no_rt = dol_data.sort_values(by=['date']).drop_duplicates(subset=['text']).sort_values(by=['Unnamed: 0'])
        
    #output
    returns = pd.DataFrame()
    returns['date'] = ret_LMT['Date']
    returns['intra_ret'] = ((ret_LMT['Close'] -  ret_LMT['Open'])/ret_LMT['Open'])
    intra_ret_SP = ((ret_SP['Close'] -  ret_SP['Open'])/ret_SP['Open'])
    returns['intra_multi'] = 3
    returns['intra_multi'][returns['intra_ret']<-0.003 ] = 2
    returns['intra_multi'][returns['intra_ret']<-0.02 ] = 1
    returns['intra_multi'][returns['intra_ret']>0.003 ] = 4
    returns['intra_multi'][returns['intra_ret']>0.02 ] = 5
    returns['intra_multi2'] = 2
    returns['intra_multi2'][returns['intra_ret']<-0.003 ] = 1
    returns['intra_multi2'][returns['intra_ret']>0.003 ] = 3
    
    
    
    #abnormal returns
    abn_ret_model = sm.OLS(returns['intra_ret'], sm.add_constant(intra_ret_SP)).fit()
    returns['abn_ret'] = returns['intra_ret'] - abn_ret_model.predict()
    
    #for the trading simulation
    trading_df_ = pd.DataFrame()
    trading_df_['date'] =  ret_LMT.loc[ret_LMT['Date']>= cutoff][ ret_LMT['Date']< cutoff2]['Date']
    trading_df_['1_day_dif'] = ((ret_LMT['Close'] -  ret_LMT['Open'])/ ret_LMT['Open']).loc[ret_LMT['Date']>= cutoff][ ret_LMT['Date']< cutoff2]
    trading_df_['abs_dif_1_day'] = ret_LMT['Close'] -  ret_LMT['Open']
    trading_df_['month'] = trading_df_['date'].str[5:7]
    trading_df_BAK = trading_df_
    
    
    #for the trading simulation 2
    trading_df2_ = pd.DataFrame()
    trading_df2_['date'] =  ret_LMT.loc[ret_LMT['Date']>= cutoff][ ret_LMT['Date']< cutoff2]['Date']
    trading_df2_['1_day_dif'] = ((ret_LMT['Close'] -  ret_LMT['Open'])/ ret_LMT['Open']).loc[ret_LMT['Date']>= cutoff][ ret_LMT['Date']< cutoff2]
    trading_df2_['abs_dif_1_day'] = ret_LMT['Close'] -  ret_LMT['Open']
    trading_df2_['month'] = trading_df2_['date'].str[5:7]
    trading_df2_BAK = trading_df2_
    
    for dataset in datasets:
        if dataset == 'dol_data':
            data = dol_data
        if dataset == 'data_full':
            data = data_full
        if dataset == 'data_full_no_rt':
            data = data_full_no_rt
        if dataset == 'dol_data_no_rt':
            data = dol_data_no_rt
        if dataset == 'data_full_FT':
            data = data_full_FT
        trading_df = trading_df_ 
        trading_df2 = trading_df2_
        for score in scores:
            model = dataset
            model = model + "_" + score
            # dates = list(set(data['date']))
            if len(dates) != len(data['date']):
                for i in dates:
                    if sum(data['date'] == i) == 0:
                        datalen = (len(data)+1)
                        data.append(pd.Series(name=datalen))
                        for col in data.columns:
                            data.at[datalen,col] = 0
                        data.at[datalen,'date'] = i
            dates.sort()
            pos_count = [0]*len(dates)
            tot_count = [0]*len(dates)
            neg_count = [0]*len(dates)
            TSV_1 = [0]*len(dates)
            sumaggr = [0]*len(dates)
            neg_aggr = pd.DataFrame()
            neg_aggr2 = [0]*len(dates)
            indexer = pd.DataFrame()
            neg_aggr['date'] = dates
            for weight_option in weight_opt:  
                model5 = model + "_" + weight_option
                if weight_option == 'non-weighted':
                    j = 0
                    for i in dates:                        
                        pos_count[j] = len(data[data[score]>0][data['date'] == i].index)
                        tot_count[j] = len(data[data['date'] == i].index)
                        sumaggr[j] = sum(data[score][data['date'] == i])
                        neg_count[j] = tot_count[j] - pos_count[j]   
                        TSV_1[j] = np.log((1+pos_count[j])/(1+neg_count[j]))
                        neg_aggr.at[j,'score'] = neg_count[j]/divide_checker(tot_count[j])
                        if datetime.strptime(i, '%Y-%m-%d').month !=1:
                            month_1 = datetime(datetime.strptime(i, '%Y-%m-%d').year,datetime.strptime(i, '%Y-%m-%d').month-1,1).strftime('%Y-%m-%d')[:7]
                        else:
                            month_1 = datetime(datetime.strptime(i, '%Y-%m-%d').year-1,12,1).strftime('%Y-%m-%d')[:7]
                        neg_aggr.at[j,'TF'] = datetime.strptime(i, '%Y-%m-%d').strftime('%Y-%m-%d')[:7]
                        try:
                            mu_neg = neg_aggr[(neg_aggr['TF'] == month_1)]['score'].mean()
                            sig_neg = np.std(neg_aggr[neg_aggr['TF'] == month_1]['score'])
                            neg_aggr2[j] = (neg_aggr.at[j,'score'] -  mu_neg)/divide_checker(sig_neg)
                        except:
                            neg_aggr2[j] = 0
                        j = j+1
                
                if weight_option == 'weighted':
                    j = 0
                    for i in dates:
                        pos_count[j] = sum(data[data[score]>0][data['date'] == i].weight)
                        tot_count[j] = sum(data[data['date'] == i].weight)
                        neg_count[j] = sum(data[data[score]<0][data['date'] == i].weight) 
                        sumaggr[j] = sum(data[score][data['date'] == i]*data['weight'][data['date'] == i])
                        neg_count[j] = tot_count[j] - pos_count[j]   
                        TSV_1[j] = np.log((1+pos_count[j])/(1+neg_count[j]))
                        neg_aggr.at[j,'score'] = neg_count[j]/divide_checker(tot_count[j])
                        if datetime.strptime(i, '%Y-%m-%d').month !=1:
                            month_1 = datetime(datetime.strptime(i, '%Y-%m-%d').year,datetime.strptime(i, '%Y-%m-%d').month-1,1).strftime('%Y-%m-%d')[:7]
                        else:
                            month_1 = datetime(datetime.strptime(i, '%Y-%m-%d').year-1,12,1).strftime('%Y-%m-%d')[:7]
                        neg_aggr.at[j,'TF'] = datetime.strptime(i, '%Y-%m-%d').strftime('%Y-%m-%d')[:7]
                        try:
                            mu_neg = neg_aggr[(neg_aggr['TF'] == month_1)]['score'].mean()
                            sig_neg = np.std(neg_aggr[neg_aggr['TF'] == month_1]['score'])
                            neg_aggr2[j] = (neg_aggr.at[j,'score'] -  mu_neg)/divide_checker(sig_neg)
                        except:
                            neg_aggr2[j] = 0
                        j = j+1
                neg_aggr2 = [0 if np.isnan(x) else x for x in neg_aggr2]
                #correlation between aggregates
                #corrdia = pd.DataFrame()
                #corrdia['TSV1'] = TSV_1
                #corrdia['TSV2'] = TSV_2
                #corrdia['neg1'] = neg_aggr['score']
                #corrdia['neg2'] = neg_aggr2
                #corrdia.corr()\
                
                
                data2 = pd.DataFrame()
                data2['date'] = dates
                data2['TSV'] = TSV_1
                data2['TSV_1'] = data2['TSV'].shift(periods=1)
                data2['TSV_2'] = data2['TSV'].shift(periods=2)
                data2['TSV_3'] = data2['TSV'].shift(periods=3)
                data2['TSV_4'] = data2['TSV'].shift(periods=4)
                data2['TSV_5'] = data2['TSV'].shift(periods=5)
                data2['TSV_6'] = data2['TSV'].shift(periods=6)
                data2['sumaggr'] = sumaggr
                data2['sumaggr_1'] = data2['sumaggr'].shift(periods=1)
                data2['sumaggr_2'] = data2['sumaggr'].shift(periods=2)
                data2['sumaggr_3'] = data2['sumaggr'].shift(periods=3)
                data2['sumaggr_4'] = data2['sumaggr'].shift(periods=4)
                data2['sumaggr_5'] = data2['sumaggr'].shift(periods=5)
                data2['sumaggr_6'] = data2['sumaggr'].shift(periods=6)
                data2['neg_aggr'] = neg_aggr['score']
                data2['neg_aggr_1'] = data2['neg_aggr'].shift(periods=1)
                data2['neg_aggr_2'] = data2['neg_aggr'].shift(periods=2)
                data2['neg_aggr_3'] = data2['neg_aggr'].shift(periods=3)
                data2['neg_aggr_4'] = data2['neg_aggr'].shift(periods=4)
                data2['neg_aggr_5'] = data2['neg_aggr'].shift(periods=5)
                data2['neg_aggr_6'] = data2['neg_aggr'].shift(periods=6)
                data2['neg_aggr2'] = neg_aggr2
                data2['neg_aggr2_1'] = data2['neg_aggr2'].shift(periods=1)
                data2['neg_aggr2_2'] = data2['neg_aggr2'].shift(periods=2)
                data2['neg_aggr2_3'] = data2['neg_aggr2'].shift(periods=3)
                data2['neg_aggr2_4'] = data2['neg_aggr2'].shift(periods=4)
                data2['neg_aggr2_5'] = data2['neg_aggr2'].shift(periods=5)
                data2['neg_aggr2_6'] = data2['neg_aggr2'].shift(periods=6)
                
                data_final = returns.merge(data2, left_on='date', right_on='date')
                for lag in lags:
                    model6 = model5 + "_" + lag
                    for output in output_set:
                        model2 = model6 + "_" + output
                        for aggr in aggregates:
                            model3 = model2 + "_" + aggr
                            X = pd.DataFrame()
                            X['lag1']  = data_final.loc[data_final['date']< cutoff][aggr+'_1'] 
                            X['lag2']  = data_final.loc[data_final['date']< cutoff][aggr+'_2']  
                            X['lag3']  = data_final.loc[data_final['date']< cutoff][aggr+'_3'] 
                            if lag in ['lags4','lags5','lags6']:
                                X['lag4']  = data_final.loc[data_final['date']< cutoff][aggr+'_4'] 
                            if lag in ['lags5','lags6']:
                                X['lag5']  = data_final.loc[data_final['date']< cutoff][aggr+'_5'] 
                            if lag == 'lags6':
                                X['lag6']  = data_final.loc[data_final['date']< cutoff][aggr+'_6'] 
                            X = X.replace([np.inf, -np.inf], np.nan).dropna()
                            
                            if output == 'abnormal':
                                y = data_final.loc[data_final['date']< cutoff]['abn_ret']
                            if output == 'intra':
                                y = data_final.loc[data_final['date']< cutoff]['intra_ret']
                            if output == 'multi':
                                y = data_final.loc[data_final['date']< cutoff]['intra_multi']
                            if output == 'multi2':
                                y = data_final.loc[data_final['date']< cutoff]['intra_multi2']    
                                
                            y = y[X.index]
                            y_dum = y
                            y_dum[y_dum<= 0] = 0
                            y_dum[y_dum> 0] = 1
                            
                            if output == 'abnormal':
                                y = data_final.loc[data_final['date']< cutoff]['abn_ret']
                            if output == 'intra':
                                y = data_final.loc[data_final['date']< cutoff]['intra_ret']
                            if output == 'multi':
                                y_dum = data_final.loc[data_final['date']< cutoff]['intra_multi']
                                y_dum = y_dum[X.index]
                                y_dum_reg = data_final.loc[data_final['date']< cutoff]['abn_ret']
                                y_dum_reg = y_dum_reg[X.index]
                            if output == 'multi2':
                                y_dum = data_final.loc[data_final['date']< cutoff]['intra_multi2']
                                y_dum = y_dum[X.index]  
                                y_dum_reg = data_final.loc[data_final['date']< cutoff]['abn_ret']
                                y_dum_reg = y_dum_reg[X.index]
                            y = y[X.index]
                            
                            
                            #y = data_final.loc[data_final['date']< cutoff]['intra_ret']
                            
                            X_test = pd.DataFrame()
                            X_test['lag1']  = data_final.loc[data_final['date']>= cutoff][ data_final['date']< cutoff2][aggr+'_1'] 
                            X_test['lag2']  = data_final.loc[data_final['date']>= cutoff][ data_final['date']< cutoff2][aggr+'_2'] 
                            X_test['lag3']  = data_final.loc[data_final['date']>= cutoff][ data_final['date']< cutoff2][aggr+'_3'] 
                            if lag in ['lags4','lags5','lags6']:
                                X_test['lag4']  = data_final.loc[data_final['date']>= cutoff][ data_final['date']< cutoff2][aggr+'_4'] 
                            if lag in ['lags5','lags6']:
                                X_test['lag5']  = data_final.loc[data_final['date']>= cutoff][ data_final['date']< cutoff2][aggr+'_5'] 
                            if lag == 'lags6':
                                X_test['lag6']  = data_final.loc[data_final['date']>= cutoff][ data_final['date']< cutoff2][aggr+'_6'] 
                            
                            if output == 'abnormal':
                               y_test = data_final.loc[data_final['date']>= cutoff][ data_final['date']< cutoff2]['abn_ret']
                            if output == 'intra':
                                y_test = data_final.loc[data_final['date']>= cutoff][ data_final['date']< cutoff2]['intra_ret']
                            if output == 'multi':
                                y_test = data_final.loc[data_final['date']>= cutoff][ data_final['date']< cutoff2]['intra_multi']
                                y_dum_reg_test = data_final.loc[data_final['date']>= cutoff]['abn_ret']
                            if output == 'multi2':
                                y_test = data_final.loc[data_final['date']>= cutoff][ data_final['date']< cutoff2]['intra_multi2']
                                y_dum_reg_test = data_final.loc[data_final['date']>= cutoff][ data_final['date']< cutoff2]['abn_ret']
                            yt_dum = y_test
                            yt_dum[yt_dum<= 0] = 0
                            yt_dum[yt_dum> 0] = 1
                            
                            if output == 'abnormal':
                               y_test = data_final.loc[data_final['date']>= cutoff][ data_final['date']< cutoff2]['abn_ret']
                            if output == 'intra':
                                y_test = data_final.loc[data_final['date']>= cutoff][ data_final['date']< cutoff2]['intra_ret']
                            if output == 'multi':
                                yt_dum = data_final.loc[data_final['date']>= cutoff][ data_final['date']< cutoff2]['intra_multi']
                            if output == 'multi2':
                                yt_dum = data_final.loc[data_final['date']>= cutoff][ data_final['date']< cutoff2]['intra_multi2']    
                            
                            #models ----------------------------------------------------------------------------------------
                            
                            #ols
                            try:
                                if output != 'multi' and output != 'multi2':
                                    OLS_model = sm.OLS(y, sm.add_constant(X.astype(float))).fit()
                                    y_pred = round((1+np.sign(OLS_model.predict(sm.add_constant(X_test)))).pow(0.0000005),0)
                                    result_matrix.at[model3+"_OLS",'hitrate'] = sum(np.sign(OLS_model.predict(sm.add_constant(X_test))) == np.sign(y_test))/len(y_test)
                                    result_matrix.at[model3+"_OLS",'ROR'] = trading_sim(y_pred,trading_df)[0]
                                    result_matrix.at[model3+"_OLS",'sharpe'] = trading_sim(y_pred,trading_df)[1]
                                    result_matrix.at[model3+"_OLS",'profit'] = trading_sim(y_pred,trading_df)[2]
                                    y_pred.index = yt_dum.index
                                    class_rep = pd.DataFrame(classification_report(y_pred,yt_dum,output_dict=True)).transpose()
                                    result_matrix.at[model3+"_OLS",'precision'] = class_rep.loc['macro avg','precision']
                                    result_matrix.at[model3+"_OLS",'recall'] = class_rep.loc['macro avg','recall']
                                    result_matrix.at[model3+"_OLS",'Auroc'] = roc_auc_score(yt_dum,y_pred)
                                if output == 'multi2':
                                    OLS_model = sm.OLS(y_dum_reg, sm.add_constant(X.astype(float))).fit()
                                    y_pred2 = OLS_model.predict(sm.add_constant(X_test))
                                    y_pred = pd.DataFrame([2] *(len(y_pred2)))
                                    y_pred.index = y_pred2.index
                                    y_pred[y_pred2<-0.003 ] = 1
                                    y_pred[y_pred2>0.003 ] = 3
                                    y_pred = y_pred.iloc[:,0]
                                    y_pred_2 = y_pred
                                    result_matrix.at[model3+"_OLS",'hitrate'] = sum(yt_dum == y_pred)/len(yt_dum)
                                    result_matrix.at[model3+"_OLS",'ROR'] = trading_sim_multi2(y_pred,trading_df)[0]
                                    result_matrix.at[model3+"_OLS",'sharpe'] = trading_sim_multi2(y_pred,trading_df)[1]
                                    result_matrix.at[model3+"_OLS",'profit'] = trading_sim_multi2(y_pred,trading_df)[2]
                                    y_pred_2.index = y_pred2.index
                                    class_rep = pd.DataFrame(classification_report(y_pred,yt_dum,output_dict=True)).transpose()
                                    result_matrix.at[model3+"_OLS",'precision'] = class_rep.loc['macro avg','precision']
                                    result_matrix.at[model3+"_OLS",'recall'] = class_rep.loc['macro avg','recall']
                                    result_matrix.at[model3+"_OLS",'adj_hr'] = 1-(sum(y_pred_2[yt_dum == 1] == 3)+ sum(y_pred_2[yt_dum == 3] == 1))/len(y_pred_2)
                                    try:
                                        result_matrix.at[model3+"_OLS",'adj_hr1'] = 1-(sum(y_pred_2[yt_dum == 1] == 3)+ sum(y_pred_2[yt_dum == 3] == 1))/sum(y_pred_2 !=2)
                                    except:
                                        result_matrix.at[model3+"_OLS",'adj_hr1'] = 0
                                    result_matrix.at[model3+"_OLS",'cohen_kappa'] = cohen_kappa_score(yt_dum,y_pred_2)
                                
                                #logistic regression
                                log_model = LogisticRegression()
                                # if output != 'multi':
                                #     log_model = LogisticRegression(multi_class = 'multinomial')
                                log_model.fit(X,y_dum)  
                                y_pred = log_model.predict(X_test)
                                if output != 'multi' and output != 'multi2':
                                    result_matrix.at[model3+"_logreg",'ROR'] = trading_sim(y_pred,trading_df)[0]
                                    result_matrix.at[model3+"_logreg",'hitrate'] = sum(y_pred == yt_dum)/len(yt_dum)
                                    result_matrix.at[model3+"_logreg",'sharpe'] = trading_sim(y_pred,trading_df)[1]
                                    result_matrix.at[model3+"_logreg",'profit'] = trading_sim(y_pred,trading_df)[2]
                                    class_rep = pd.DataFrame(classification_report(y_pred,yt_dum,output_dict=True)).transpose()
                                    result_matrix.at[model3+"_logreg",'precision'] = class_rep.loc['macro avg','precision']
                                    result_matrix.at[model3+"_logreg",'recall'] = class_rep.loc['macro avg','recall']
                                    result_matrix.at[model3+"_logreg",'Auroc'] = roc_auc_score(yt_dum,y_pred)
                                if output == 'multi':
                                    result_matrix.at[model3+"_logreg",'ROR'] = trading_sim_multi(y_pred,trading_df)[0]
                                    result_matrix.at[model3+"_logreg",'hitrate'] = sum(y_pred == yt_dum)/len(yt_dum)
                                    result_matrix.at[model3+"_logreg",'sharpe'] = trading_sim_multi(y_pred,trading_df)[1]
                                    result_matrix.at[model3+"_logreg",'profit'] = trading_sim_multi(y_pred,trading_df)[2]
                                    class_rep = pd.DataFrame(classification_report(y_pred,yt_dum,output_dict=True)).transpose()
                                    result_matrix.at[model3+"_logreg",'precision'] = class_rep.loc['macro avg','precision']
                                    result_matrix.at[model3+"_logreg",'recall'] = class_rep.loc['macro avg','recall']
                                    result_matrix.at[model3+"_logreg",'adj_hr'] = 1-(sum(y_pred[yt_dum == 1] == 5)+ sum(y_pred[yt_dum == 1] == 4)+sum(y_pred[yt_dum == 2] == 5)+sum(y_pred[yt_dum == 2] == 4)+sum(y_pred[yt_dum == 4] == 1)+sum(y_pred[yt_dum == 4] == 2)+sum(y_pred[yt_dum == 5] == 1)+sum(y_pred[yt_dum == 5] == 2))/len(y_pred)
                                    result_matrix.at[model3+"_logreg",'adj_hr1'] = 1-(sum(y_pred[yt_dum == 1] == 5)+ sum(y_pred[yt_dum == 1] == 4)+sum(y_pred[yt_dum == 2] == 5)+sum(y_pred[yt_dum == 2] == 4)+sum(y_pred[yt_dum == 4] == 1)+sum(y_pred[yt_dum == 4] == 2)+sum(y_pred[yt_dum == 5] == 1)+sum(y_pred[yt_dum == 5] == 2))/sum(y_pred !=3)
                                    result_matrix.at[model3+"_logreg",'cohen_kappa'] = cohen_kappa_score(yt_dum,y_pred)
                                if output == 'multi2':
                                    result_matrix.at[model3+"_logreg",'ROR'] = trading_sim_multi2(y_pred,trading_df)[0]
                                    result_matrix.at[model3+"_logreg",'hitrate'] = sum(y_pred == yt_dum)/len(yt_dum)
                                    result_matrix.at[model3+"_logreg",'sharpe'] = trading_sim_multi2(y_pred,trading_df)[1]
                                    result_matrix.at[model3+"_logreg",'profit'] = trading_sim_multi2(y_pred,trading_df)[2]
                                    class_rep = pd.DataFrame(classification_report(y_pred,yt_dum,output_dict=True)).transpose()
                                    result_matrix.at[model3+"_logreg",'precision'] = class_rep.loc['macro avg','precision']
                                    result_matrix.at[model3+"_logreg",'recall'] = class_rep.loc['macro avg','recall']
                                    result_matrix.at[model3+"_logreg",'adj_hr'] = 1-(sum(y_pred[yt_dum == 1] == 3)+ sum(y_pred[yt_dum == 3] == 1))/len(y_pred)
                                    result_matrix.at[model3+"_logreg",'adj_hr1'] = 1-(sum(y_pred[yt_dum == 1] == 3)+ sum(y_pred[yt_dum == 3] == 1))/sum(y_pred !=2)
                                    result_matrix.at[model3+"_logreg",'cohen_kappa'] = cohen_kappa_score(yt_dum,y_pred)    
                                    
                                #support vector machine
                                SVM = svm.SVC(max_iter = 10000)
                                # if output != 'multi':  
                                #     SVM = svm.SVC(decision_function_shape='ovr')
                                SVM.fit(X, y_dum)
                                y_pred = SVM.predict(X_test)
                                if output != 'multi' and output != 'multi2':
                                    result_matrix.at[model3+"_SVM",'ROR'] = trading_sim(y_pred,trading_df)[0]
                                    result_matrix.at[model3+"_SVM",'hitrate'] = sum(y_pred == yt_dum)/len(yt_dum)
                                    result_matrix.at[model3+"_SVM",'sharpe'] = trading_sim(y_pred,trading_df)[1]
                                    result_matrix.at[model3+"_SVM",'profit'] = trading_sim(y_pred,trading_df)[2]
                                    class_rep = pd.DataFrame(classification_report(y_pred,yt_dum,output_dict=True)).transpose()
                                    result_matrix.at[model3+"_SVM",'precision'] = class_rep.loc['macro avg','precision']
                                    result_matrix.at[model3+"_SVM",'recall'] = class_rep.loc['macro avg','recall']
                                    result_matrix.at[model3+"_SVM",'Auroc'] = roc_auc_score(yt_dum,y_pred)
                                if output == 'multi':
                                    result_matrix.at[model3+"_SVM",'ROR'] = trading_sim_multi(y_pred,trading_df)[0]
                                    result_matrix.at[model3+"_SVM",'hitrate'] = sum(y_pred == yt_dum)/len(yt_dum)
                                    result_matrix.at[model3+"_SVM",'sharpe'] = trading_sim_multi(y_pred,trading_df)[1]
                                    result_matrix.at[model3+"_SVM",'profit'] = trading_sim_multi(y_pred,trading_df)[2]
                                    class_rep = pd.DataFrame(classification_report(y_pred,yt_dum,output_dict=True)).transpose()
                                    result_matrix.at[model3+"_SVM",'precision'] = class_rep.loc['macro avg','precision']
                                    result_matrix.at[model3+"_SVM",'recall'] = class_rep.loc['macro avg','recall']
                                    result_matrix.at[model3+"_SVM",'adj_hr'] = 1-(sum(y_pred[yt_dum == 1] == 5)+ sum(y_pred[yt_dum == 1] == 4)+sum(y_pred[yt_dum == 2] == 5)+sum(y_pred[yt_dum == 2] == 4)+sum(y_pred[yt_dum == 4] == 1)+sum(y_pred[yt_dum == 4] == 2)+sum(y_pred[yt_dum == 5] == 1)+sum(y_pred[yt_dum == 5] == 2))/len(y_pred)
                                    result_matrix.at[model3+"_SVM",'adj_hr1'] = 1-(sum(y_pred[yt_dum == 1] == 5)+ sum(y_pred[yt_dum == 1] == 4)+sum(y_pred[yt_dum == 2] == 5)+sum(y_pred[yt_dum == 2] == 4)+sum(y_pred[yt_dum == 4] == 1)+sum(y_pred[yt_dum == 4] == 2)+sum(y_pred[yt_dum == 5] == 1)+sum(y_pred[yt_dum == 5] == 2))/sum(y_pred !=3)
                                    result_matrix.at[model3+"_SVM",'cohen_kappa'] = cohen_kappa_score(yt_dum,y_pred)
                                if output == 'multi2':
                                    result_matrix.at[model3+"_SVM",'ROR'] = trading_sim_multi2(y_pred,trading_df)[0]
                                    result_matrix.at[model3+"_SVM",'hitrate'] = sum(y_pred == yt_dum)/len(yt_dum)
                                    result_matrix.at[model3+"_SVM",'sharpe'] = trading_sim_multi2(y_pred,trading_df)[1]
                                    result_matrix.at[model3+"_SVM",'profit'] = trading_sim_multi2(y_pred,trading_df)[2]
                                    class_rep = pd.DataFrame(classification_report(y_pred,yt_dum,output_dict=True)).transpose()
                                    result_matrix.at[model3+"_SVM",'precision'] = class_rep.loc['macro avg','precision']
                                    result_matrix.at[model3+"_SVM",'recall'] = class_rep.loc['macro avg','recall']
                                    result_matrix.at[model3+"_SVM",'adj_hr'] = 1-(sum(y_pred[yt_dum == 1] == 3)+ sum(y_pred[yt_dum == 3] == 1))/len(y_pred)
                                    result_matrix.at[model3+"_SVM",'adj_hr1'] = 1-(sum(y_pred[yt_dum == 1] == 3)+ sum(y_pred[yt_dum == 3] == 1))/sum(y_pred !=2)
                                    result_matrix.at[model3+"_SVM",'cohen_kappa'] = cohen_kappa_score(yt_dum,y_pred)    
                                    
                                #support vector regression
                                if output != 'multi' and output != 'multi2':
                                    SVR_model = svm.SVR(kernel='rbf',max_iter = 10000)
                                    min_max_scaler = preprocessing.MinMaxScaler()
                                    x_scaled = min_max_scaler.fit_transform(X)
                                    min_max_scaler = preprocessing.MinMaxScaler()
                                    y_norm = (y-min(y))/(max(y)-min(y))
                                    x_test_scaled = min_max_scaler.fit_transform(X_test)
                                    SVR_model.fit(x_scaled, y_norm)
                                    y_pred = ((SVR_model.predict(x_test_scaled))*(max(y)-min(y))+min(y))
                                    y_pred[y_pred>0] = 1
                                    y_pred[y_pred<=0] = 0
                                    result_matrix.at[model3+"_SVR",'ROR'] = trading_sim(y_pred,trading_df)[0]
                                    result_matrix.at[model3+"_SVR",'hitrate'] = sum(y_pred == yt_dum)/len(yt_dum)
                                    result_matrix.at[model3+"_SVR",'sharpe'] = trading_sim(y_pred,trading_df)[1]
                                    result_matrix.at[model3+"_SVR",'profit'] = trading_sim_multi2(y_pred,trading_df)[2]
                                    class_rep = pd.DataFrame(classification_report(y_pred,yt_dum,output_dict=True)).transpose()
                                    result_matrix.at[model3+"_SVR",'precision'] = class_rep.loc['macro avg','precision']
                                    result_matrix.at[model3+"_SVR",'recall'] = class_rep.loc['macro avg','recall']
                                    result_matrix.at[model3+"_SVR",'Auroc'] = roc_auc_score(yt_dum,y_pred)
                                if output == 'multi2':
                                    SVR_model = svm.SVR(kernel='rbf',max_iter = 10000)
                                    min_max_scaler = preprocessing.MinMaxScaler()
                                    x_scaled = min_max_scaler.fit_transform(X)
                                    min_max_scaler = preprocessing.MinMaxScaler()
                                    y_norm = (y_dum_reg-min(y_dum_reg))/(max(y_dum_reg)-min(y_dum_reg))
                                    x_test_scaled = min_max_scaler.fit_transform(X_test)
                                    SVR_model.fit(x_scaled, y_norm)
                                    y_pred2 = ((SVR_model.predict(x_test_scaled))*(max(y_dum_reg)-min(y_dum_reg))+min(y_dum_reg))
                                    y_pred2 = pd.DataFrame(y_pred2)
                                    y_pred = pd.DataFrame([2]* len(y_pred))
                                    y_pred[y_pred2<-0.003 ] = 1
                                    y_pred[y_pred2>0.003 ] = 3
                                    y_pred = y_pred.iloc[:,0]
                                    y_pred.index = yt_dum.index
                                    y_pred_2 = y_pred
                                    result_matrix.at[model3+"_SVR",'hitrate'] = sum(y_pred == yt_dum)/len(yt_dum)
                                    result_matrix.at[model3+"_SVR",'ROR'] = trading_sim_multi2(y_pred,trading_df)[0]
                                    result_matrix.at[model3+"_SVR",'sharpe'] = trading_sim_multi2(y_pred,trading_df)[1]
                                    result_matrix.at[model3+"_SVR",'profit'] = trading_sim_multi2(y_pred,trading_df)[2]
                                    y_pred_2.index = yt_dum.index
                                    class_rep = pd.DataFrame(classification_report(y_pred,yt_dum,output_dict=True)).transpose()
                                    result_matrix.at[model3+"_SVR",'recall'] = class_rep.loc['macro avg','recall']
                                    result_matrix.at[model3+"_SVR",'precision'] = class_rep.loc['macro avg','precision']
                                    result_matrix.at[model3+"_SVR",'adj_hr'] = 1-(sum(y_pred_2[yt_dum == 1] == 3)+ sum(y_pred_2[yt_dum == 3] == 1))/len(y_pred_2)
                                    try:
                                        result_matrix.at[model3+"_SVR",'adj_hr1'] = 1-(sum(y_pred_2[yt_dum == 1] == 3)+ sum(y_pred_2[yt_dum == 3] == 1))/sum(y_pred_2 !=2)
                                    except:
                                        result_matrix.at[model3+"_SVR",'adj_hr1'] = 0
                                    result_matrix.at[model3+"_SVR",'cohen_kappa'] = cohen_kappa_score(yt_dum,y_pred_2)
                                
                                #Naive Bayes Classifier
                                if output != 'multi' and output != 'multi2':
                                    bnb = BernoulliNB()
                                    y_pred = bnb.fit(X, y_dum).predict(X_test)
                                    result_matrix.at[model3+"_NB",'ROR'] = trading_sim(y_pred,trading_df)[0]
                                    result_matrix.at[model3+"_NB",'hitrate'] = sum(y_pred == yt_dum)/len(yt_dum)
                                    result_matrix.at[model3+"_NB",'sharpe'] = trading_sim(y_pred,trading_df)[1]
                                    result_matrix.at[model3+"_NB",'profit'] = trading_sim(y_pred,trading_df)[2]
                                    class_rep = pd.DataFrame(classification_report(y_pred,yt_dum,output_dict=True)).transpose()
                                    result_matrix.at[model3+"_NB",'precision'] = class_rep.loc['macro avg','precision']
                                    result_matrix.at[model3+"_NB",'recall'] = class_rep.loc['macro avg','recall']
                                    result_matrix.at[model3+"_NB",'Auroc'] = roc_auc_score(yt_dum,y_pred)
                                if output == 'multi':
                                    gnb = GaussianNB()
                                    y_pred = gnb.fit(X, y_dum).predict(X_test)
                                    result_matrix.at[model3+"_NB",'ROR'] = trading_sim_multi(y_pred,trading_df)[0]
                                    result_matrix.at[model3+"_NB",'hitrate'] = sum(y_pred == yt_dum)/len(yt_dum)
                                    result_matrix.at[model3+"_NB",'sharpe'] = trading_sim_multi(y_pred,trading_df)[1]
                                    result_matrix.at[model3+"_NB",'profit'] = trading_sim_multi(y_pred,trading_df)[2]
                                    class_rep = pd.DataFrame(classification_report(y_pred,yt_dum,output_dict=True)).transpose()
                                    result_matrix.at[model3+"_NB",'precision'] = class_rep.loc['macro avg','precision']
                                    result_matrix.at[model3+"_NB",'recall'] = class_rep.loc['macro avg','recall']
                                    result_matrix.at[model3+"_NB",'adj_hr'] = 1-(sum(y_pred[yt_dum == 1] == 5)+ sum(y_pred[yt_dum == 1] == 4)+sum(y_pred[yt_dum == 2] == 5)+sum(y_pred[yt_dum == 2] == 4)+sum(y_pred[yt_dum == 4] == 1)+sum(y_pred[yt_dum == 4] == 2)+sum(y_pred[yt_dum == 5] == 1)+sum(y_pred[yt_dum == 5] == 2))/len(y_pred)
                                    result_matrix.at[model3+"_NB",'adj_hr1'] = 1-(sum(y_pred[yt_dum == 1] == 5)+ sum(y_pred[yt_dum == 1] == 4)+sum(y_pred[yt_dum == 2] == 5)+sum(y_pred[yt_dum == 2] == 4)+sum(y_pred[yt_dum == 4] == 1)+sum(y_pred[yt_dum == 4] == 2)+sum(y_pred[yt_dum == 5] == 1)+sum(y_pred[yt_dum == 5] == 2))/sum(y_pred !=3)
                                    result_matrix.at[model3+"_NB",'cohen_kappa'] = cohen_kappa_score(yt_dum,y_pred)
                                if output == 'multi2':
                                    gnb = GaussianNB()
                                    y_pred = gnb.fit(X, y_dum).predict(X_test)
                                    result_matrix.at[model3+"_NB",'ROR'] = trading_sim_multi2(y_pred,trading_df)[0]
                                    result_matrix.at[model3+"_NB",'hitrate'] = sum(y_pred == yt_dum)/len(yt_dum)
                                    result_matrix.at[model3+"_NB",'sharpe'] = trading_sim_multi2(y_pred,trading_df)[1]
                                    result_matrix.at[model3+"_NB",'profit'] = trading_sim_multi2(y_pred,trading_df)[2]
                                    class_rep = pd.DataFrame(classification_report(y_pred,yt_dum,output_dict=True)).transpose()
                                    result_matrix.at[model3+"_NB",'precision'] = class_rep.loc['macro avg','precision']
                                    result_matrix.at[model3+"_NB",'recall'] = class_rep.loc['macro avg','recall']
                                    result_matrix.at[model3+"_NB",'adj_hr'] = 1-(sum(y_pred[yt_dum == 1] == 3)+ sum(y_pred[yt_dum == 3] == 1))/len(y_pred)
                                    result_matrix.at[model3+"_NB",'adj_hr1'] = 1-(sum(y_pred[yt_dum == 1] == 3)+ sum(y_pred[yt_dum == 3] == 1))/sum(y_pred !=2)
                                    result_matrix.at[model3+"_NB",'cohen_kappa'] = cohen_kappa_score(yt_dum,y_pred)    
                            except:
                                pass
                            print(str(round(nr/length,2)*100) + " % completed")
                            nr = nr+1
    # result_matrix = pd.read_csv("Results_HAS.csv")
    y = result_matrix['sharpe'].replace([np.inf, -np.inf], np.nan).dropna(axis=0).sort_values(ascending=False)[0:10].index
    res_section1[company] = result_matrix.loc[y].index


res_section1.to_csv("res_section1.csv")
    

#----------------------------------------------- SECTION 2 ----------------------------------------------------------------------- 
# For each of the models the decision made by the models chosen in section 1 over the first 2 years of data are extracted.
res_section1 = pd.read_csv("res_section1.csv")
result_matrix = pd.DataFrame()
LMT_res_section2 = pd.DataFrame()
HAS_res_section2 = pd.DataFrame()
NVIDIA_res_section2 = pd.DataFrame()
nr = 1
dates = dol_data['date'].unique()
for company in companies:
    for topmodel in res_section1[company]:
        if 'TSV' in topmodel:
            aggregates = ['TSV']
        if 'neg_aggr' in topmodel:
            aggregates = ['neg_aggr']
        if 'neg_aggr2' in topmodel:
            aggregates = ['neg_aggr2']
        if 'sumaggr' in topmodel:
            aggregates = ['sumaggr']
        if 'afinn_score' in topmodel:
            scores = ['afinn_score']
        if 'senti_score' in topmodel:
            scores = ['senti_score']
        if 'OpFi_score' in topmodel:
            scores = ['OpFi_score']
        if 'emoji' in topmodel:
            scores = ['emoji']
        if 'SWN' in topmodel:
            scores = ['SWN']
        if 'intra' in topmodel:
            output_set = ['intra']
        if 'abnormal' in topmodel:
            output_set = ['abnormal']
        if 'multi' in topmodel:
            output_set = ['multi']
        if 'multi2' in topmodel:
            output_set = ['multi2']
        if 'data_full' in topmodel:
            datasets = ['data_full']
        if 'dol_data' in topmodel:
            datasets = ['dol_data']
        if 'data_full_FT' in topmodel:
            datasets = ['data_full_FT']
        if 'dol_data_no_rt' in topmodel:
            datasets = ['dol_data_no_rt']
        if 'data_full_no_rt' in topmodel:
            datasets = ['data_full_no_rt']
        if 'weighted' in topmodel:
            weight_opt = ['weighted']
        if 'non-weighted' in topmodel:
            weight_opt = ['non-weighted']
        if 'lags3' in topmodel:
            lags = ['lags3']
        if 'lags4' in topmodel:
            lags = ['lags4']
        if 'lags5' in topmodel:
            lags = ['lags5']
        if 'lags6' in topmodel:
            lags = ['lags6']
            
        #LMT
        if company == 'LMT':
            if 'data_full' in topmodel:
                data_full = pd.read_csv("LMTdata2016-2018_rev5.csv") 
                data_full['senti_score'] = data_full['senti_score_pos'] + data_full['senti_score_neg']
            if 'dol_data' in topmodel:
                dol_data = pd.read_csv("Dollar_LMT_data2016-2018_rev5.csv")
                dol_data['senti_score'] = dol_data['senti_score_pos'] + dol_data['senti_score_neg']
            if 'data_full_FT' in topmodel:
                data_full_FT = pd.read_csv("LMT_FT_Data2016-2018.csv")
                data_full_FT['senti_score'] = dol_data['senti_score_pos'] + dol_data['senti_score_neg']
            if 'data_full_no_rt' in topmodel:
                data_full = pd.read_csv("LMTdata2016-2018_rev5.csv") 
                data_full['senti_score'] = data_full['senti_score_pos'] + data_full['senti_score_neg']
                data_full_no_rt = data_full.sort_values(by=['date']).drop_duplicates(subset=['text']).sort_values(by=['Unnamed: 0'])
            if 'dol_data_no_rt' in topmodel:
                dol_data = pd.read_csv("Dollar_LMT_data2016-2018_rev5.csv")
                dol_data['senti_score'] = dol_data['senti_score_pos'] + dol_data['senti_score_neg']
                dol_data_no_rt = dol_data.sort_values(by=['date']).drop_duplicates(subset=['text']).sort_values(by=['Unnamed: 0'])
            ret_SP = pd.read_csv("^GSPC.csv") 
            ret_LMT = pd.read_csv("LMT.csv")
            
        #NVIDIA
        if company == 'NVIDIA':
            if 'data_full' in topmodel:
                data_full = pd.read_csv("NVIDIAdata2016-2018_rev5_2.csv") 
                data_full['senti_score'] = data_full['senti_score_pos'] + data_full['senti_score_neg']
            if 'dol_data' in topmodel:
                dol_data = pd.read_csv("dNVIDIAdata2016-2018_rev5_2.csv")
                dol_data['senti_score'] = dol_data['senti_score_pos'] + dol_data['senti_score_neg']
            if 'data_full_FT' in topmodel:
                data_full_FT = pd.read_csv("NVIDIA_FT_Data2016-2018.csv")
                data_full_FT['senti_score'] = dol_data['senti_score_pos'] + dol_data['senti_score_neg']
            if 'data_full_no_rt' in topmodel:
                data_full = pd.read_csv("NVIDIAdata2016-2018_rev5_2.csv") 
                data_full['senti_score'] = data_full['senti_score_pos'] + data_full['senti_score_neg']
                data_full_no_rt = data_full.sort_values(by=['date']).drop_duplicates(subset=['text']).sort_values(by=['Unnamed: 0'])
            if 'dol_data_no_rt' in topmodel:
                dol_data = pd.read_csv("dNVIDIAdata2016-2018_rev5_2.csv")
                dol_data['senti_score'] = dol_data['senti_score_pos'] + dol_data['senti_score_neg']
                dol_data_no_rt = dol_data.sort_values(by=['date']).drop_duplicates(subset=['text']).sort_values(by=['Unnamed: 0'])
            ret_SP = pd.read_csv("^GSPC.csv") 
            ret_LMT = pd.read_csv("NVDA.csv")
        
        #HAS
        if company == 'HAS':
            if 'data_full' in topmodel:
                data_full = pd.read_csv("HASdata2016-2018_rev5_2.csv") 
                data_full['senti_score'] = data_full['senti_score_pos'] + data_full['senti_score_neg']
            if 'dol_data' in topmodel:
                dol_data = pd.read_csv("dHASdata2016-2018_rev5.csv")
                dol_data['senti_score'] = dol_data['senti_score_pos'] + dol_data['senti_score_neg']
            if 'data_full_FT' in topmodel:
                data_full_FT = pd.read_csv("HAS_FT_Data2016-2018.csv")
                data_full_FT['senti_score'] = dol_data['senti_score_pos'] + dol_data['senti_score_neg']
            if 'data_full_no_rt' in topmodel:
                data_full = pd.read_csv("HASdata2016-2018_rev5_2.csv") 
                data_full['senti_score'] = data_full['senti_score_pos'] + data_full['senti_score_neg']
                data_full_no_rt = data_full.sort_values(by=['date']).drop_duplicates(subset=['text']).sort_values(by=['Unnamed: 0'])
            if 'dol_data_no_rt' in topmodel:
                dol_data = pd.read_csv("dHASdata2016-2018_rev5.csv")
                dol_data['senti_score'] = dol_data['senti_score_pos'] + dol_data['senti_score_neg']
                dol_data_no_rt = dol_data.sort_values(by=['date']).drop_duplicates(subset=['text']).sort_values(by=['Unnamed: 0'])
            ret_SP = pd.read_csv("^GSPC.csv") 
            ret_LMT = pd.read_csv("HAS.csv")
    
            
        #output
        returns = pd.DataFrame()
        returns['date'] = ret_LMT['Date']
        returns['intra_ret'] = ((ret_LMT['Close'] -  ret_LMT['Open'])/ret_LMT['Open'])
        intra_ret_SP = ((ret_SP['Close'] -  ret_SP['Open'])/ret_SP['Open'])
        returns['intra_multi'] = 3
        returns['intra_multi'][returns['intra_ret']<-0.003 ] = 2
        returns['intra_multi'][returns['intra_ret']<-0.02 ] = 1
        returns['intra_multi'][returns['intra_ret']>0.003 ] = 4
        returns['intra_multi'][returns['intra_ret']>0.02 ] = 5
        returns['intra_multi2'] = 2
        returns['intra_multi2'][returns['intra_ret']<-0.003 ] = 1
        returns['intra_multi2'][returns['intra_ret']>0.003 ] = 3
        
        
        
        #abnormal returns
        abn_ret_model = sm.OLS(returns['intra_ret'], sm.add_constant(intra_ret_SP)).fit()
        returns['abn_ret'] = returns['intra_ret'] - abn_ret_model.predict()
        
        #for the trading simulation
        trading_df_ = pd.DataFrame()
        trading_df_['date'] =  ret_LMT.loc[ret_LMT['Date']>= cutoff2][ ret_LMT['Date']< cutoff3]['Date']
        trading_df_['1_day_dif'] = ((ret_LMT['Close'] -  ret_LMT['Open'])/ ret_LMT['Open']).loc[ret_LMT['Date']>= cutoff2][ ret_LMT['Date']< cutoff3]
        trading_df_['abs_dif_1_day'] = ret_LMT['Close'] -  ret_LMT['Open']
        trading_df_['month'] = trading_df_['date'].str[5:7]
        trading_df_BAK = trading_df_
        
        
        #for the trading simulation 2
        trading_df2_ = pd.DataFrame()
        trading_df2_['date'] =  ret_LMT.loc[ret_LMT['Date']>= cutoff2][ ret_LMT['Date']< cutoff3]['Date']
        trading_df2_['1_day_dif'] = ((ret_LMT['Close'] -  ret_LMT['Open'])/ ret_LMT['Open']).loc[ret_LMT['Date']>= cutoff2][ ret_LMT['Date']< cutoff3]
        trading_df2_['abs_dif_1_day'] = ret_LMT['Close'] -  ret_LMT['Open']
        trading_df2_['month'] = trading_df2_['date'].str[5:7]
        trading_df2_BAK = trading_df2_
        
        for dataset in datasets:
            if dataset == 'dol_data':
                data = dol_data
            if dataset == 'data_full':
                data = data_full
            if dataset == 'data_full_no_rt':
                data = data_full_no_rt
            if dataset == 'dol_data_no_rt':
                data = dol_data_no_rt
            if dataset == 'data_full_FT':
                data = data_full_FT
            trading_df = trading_df_ 
            trading_df2 = trading_df2_
            for score in scores:
                model = dataset
                model = model + "_" + score
                # dates = list(set(data['date']))
                if len(dates) != len(data['date']):
                    for i in dates:
                        if sum(data['date'] == i) == 0:
                            datalen = (len(data)+1)
                            data.append(pd.Series(name=datalen))
                            for col in data.columns:
                                data.at[datalen,col] = 0
                            data.at[datalen,'date'] = i
                dates.sort()
                pos_count = [0]*len(dates)
                tot_count = [0]*len(dates)
                neg_count = [0]*len(dates)
                TSV_1 = [0]*len(dates)
                sumaggr = [0]*len(dates)
                neg_aggr = pd.DataFrame()
                neg_aggr2 = [0]*len(dates)
                indexer = pd.DataFrame()
                neg_aggr['date'] = dates
                for weight_option in weight_opt:  
                    model5 = model + "_" + weight_option
                    if weight_option == 'non-weighted':
                        j = 0
                        for i in dates:                        
                            pos_count[j] = len(data[data[score]>0][data['date'] == i].index)
                            tot_count[j] = len(data[data['date'] == i].index)
                            sumaggr[j] = sum(data[score][data['date'] == i])
                            neg_count[j] = tot_count[j] - pos_count[j]   
                            TSV_1[j] = np.log((1+pos_count[j])/(1+neg_count[j]))
                            neg_aggr.at[j,'score'] = neg_count[j]/divide_checker(tot_count[j])
                            if datetime.strptime(i, '%Y-%m-%d').month !=1:
                                month_1 = datetime(datetime.strptime(i, '%Y-%m-%d').year,datetime.strptime(i, '%Y-%m-%d').month-1,1).strftime('%Y-%m-%d')[:7]
                            else:
                                month_1 = datetime(datetime.strptime(i, '%Y-%m-%d').year-1,12,1).strftime('%Y-%m-%d')[:7]
                            neg_aggr.at[j,'TF'] = datetime.strptime(i, '%Y-%m-%d').strftime('%Y-%m-%d')[:7]
                            try:
                                mu_neg = neg_aggr[(neg_aggr['TF'] == month_1)]['score'].mean()
                                sig_neg = np.std(neg_aggr[neg_aggr['TF'] == month_1]['score'])
                                neg_aggr2[j] = (neg_aggr.at[j,'score'] -  mu_neg)/divide_checker(sig_neg)
                            except:
                                neg_aggr2[j] = 0
                            j = j+1
                    
                    if weight_option == 'weighted':
                        j = 0
                        for i in dates:
                            pos_count[j] = sum(data[data[score]>0][data['date'] == i].weight)
                            tot_count[j] = sum(data[data['date'] == i].weight)
                            neg_count[j] = sum(data[data[score]<0][data['date'] == i].weight) 
                            sumaggr[j] = sum(data[score][data['date'] == i]*data['weight'][data['date'] == i])
                            neg_count[j] = tot_count[j] - pos_count[j]   
                            TSV_1[j] = np.log((1+pos_count[j])/(1+neg_count[j]))
                            neg_aggr.at[j,'score'] = neg_count[j]/divide_checker(tot_count[j])
                            if datetime.strptime(i, '%Y-%m-%d').month !=1:
                                month_1 = datetime(datetime.strptime(i, '%Y-%m-%d').year,datetime.strptime(i, '%Y-%m-%d').month-1,1).strftime('%Y-%m-%d')[:7]
                            else:
                                month_1 = datetime(datetime.strptime(i, '%Y-%m-%d').year-1,12,1).strftime('%Y-%m-%d')[:7]
                            neg_aggr.at[j,'TF'] = datetime.strptime(i, '%Y-%m-%d').strftime('%Y-%m-%d')[:7]
                            try:
                                mu_neg = neg_aggr[(neg_aggr['TF'] == month_1)]['score'].mean()
                                sig_neg = np.std(neg_aggr[neg_aggr['TF'] == month_1]['score'])
                                neg_aggr2[j] = (neg_aggr.at[j,'score'] -  mu_neg)/divide_checker(sig_neg)
                            except:
                                neg_aggr2[j] = 0
                            j = j+1
                    neg_aggr2 = [0 if np.isnan(x) else x for x in neg_aggr2]
                    #correlation between aggregates
                    #corrdia = pd.DataFrame()
                    #corrdia['TSV1'] = TSV_1
                    #corrdia['TSV2'] = TSV_2
                    #corrdia['neg1'] = neg_aggr['score']
                    #corrdia['neg2'] = neg_aggr2
                    #corrdia.corr()\
                    
                    
                    data2 = pd.DataFrame()
                    data2['date'] = dates
                    data2['TSV'] = TSV_1
                    data2['TSV_1'] = data2['TSV'].shift(periods=1)
                    data2['TSV_2'] = data2['TSV'].shift(periods=2)
                    data2['TSV_3'] = data2['TSV'].shift(periods=3)
                    data2['TSV_4'] = data2['TSV'].shift(periods=4)
                    data2['TSV_5'] = data2['TSV'].shift(periods=5)
                    data2['TSV_6'] = data2['TSV'].shift(periods=6)
                    data2['sumaggr'] = sumaggr
                    data2['sumaggr_1'] = data2['sumaggr'].shift(periods=1)
                    data2['sumaggr_2'] = data2['sumaggr'].shift(periods=2)
                    data2['sumaggr_3'] = data2['sumaggr'].shift(periods=3)
                    data2['sumaggr_4'] = data2['sumaggr'].shift(periods=4)
                    data2['sumaggr_5'] = data2['sumaggr'].shift(periods=5)
                    data2['sumaggr_6'] = data2['sumaggr'].shift(periods=6)
                    data2['neg_aggr'] = neg_aggr['score']
                    data2['neg_aggr_1'] = data2['neg_aggr'].shift(periods=1)
                    data2['neg_aggr_2'] = data2['neg_aggr'].shift(periods=2)
                    data2['neg_aggr_3'] = data2['neg_aggr'].shift(periods=3)
                    data2['neg_aggr_4'] = data2['neg_aggr'].shift(periods=4)
                    data2['neg_aggr_5'] = data2['neg_aggr'].shift(periods=5)
                    data2['neg_aggr_6'] = data2['neg_aggr'].shift(periods=6)
                    data2['neg_aggr2'] = neg_aggr2
                    data2['neg_aggr2_1'] = data2['neg_aggr2'].shift(periods=1)
                    data2['neg_aggr2_2'] = data2['neg_aggr2'].shift(periods=2)
                    data2['neg_aggr2_3'] = data2['neg_aggr2'].shift(periods=3)
                    data2['neg_aggr2_4'] = data2['neg_aggr2'].shift(periods=4)
                    data2['neg_aggr2_5'] = data2['neg_aggr2'].shift(periods=5)
                    data2['neg_aggr2_6'] = data2['neg_aggr2'].shift(periods=6)
                    
                    data_final = returns.merge(data2, left_on='date', right_on='date')
                    for lag in lags:
                        model6 = model5 + "_" + lag
                        for output in output_set:
                            model2 = model6 + "_" + output
                            for aggr in aggregates:
                                model3 = model2 + "_" + aggr
                                X = pd.DataFrame()
                                X['lag1']  = data_final.loc[data_final['date']< cutoff2][aggr+'_1'] 
                                X['lag2']  = data_final.loc[data_final['date']< cutoff2][aggr+'_2']  
                                X['lag3']  = data_final.loc[data_final['date']< cutoff2][aggr+'_3'] 
                                if lag in ['lags4','lags5','lags6']:
                                    X['lag4']  = data_final.loc[data_final['date']< cutoff2][aggr+'_4'] 
                                if lag in ['lags5','lags6']:
                                    X['lag5']  = data_final.loc[data_final['date']< cutoff2][aggr+'_5'] 
                                if lag == 'lags6':
                                    X['lag6']  = data_final.loc[data_final['date']< cutoff2][aggr+'_6'] 
                                X = X.replace([np.inf, -np.inf], np.nan).dropna()
                                
                                if output == 'abnormal':
                                    y = data_final.loc[data_final['date']< cutoff2]['abn_ret']
                                if output == 'intra':
                                    y = data_final.loc[data_final['date']< cutoff2]['intra_ret']
                                if output == 'multi':
                                    y = data_final.loc[data_final['date']< cutoff2]['intra_multi']
                                if output == 'multi2':
                                    y = data_final.loc[data_final['date']< cutoff2]['intra_multi2']    
                                    
                                y = y[X.index]
                                y_dum = y
                                y_dum[y_dum<= 0] = 0
                                y_dum[y_dum> 0] = 1
                                
                                if output == 'abnormal':
                                    y = data_final.loc[data_final['date']< cutoff2]['abn_ret']
                                if output == 'intra':
                                    y = data_final.loc[data_final['date']< cutoff2]['intra_ret']
                                if output == 'multi':
                                    y_dum = data_final.loc[data_final['date']< cutoff2]['intra_multi']
                                    y_dum = y_dum[X.index]
                                    y_dum_reg = data_final.loc[data_final['date']< cutoff2]['abn_ret']
                                    y_dum_reg = y_dum_reg[X.index]
                                if output == 'multi2':
                                    y_dum = data_final.loc[data_final['date']< cutoff2]['intra_multi2']
                                    y_dum = y_dum[X.index]  
                                    y_dum_reg = data_final.loc[data_final['date']< cutoff2]['abn_ret']
                                    y_dum_reg = y_dum_reg[X.index]
                                y = y[X.index]
                                
                                
                                #y = data_final.loc[data_final['date']< cutoff]['intra_ret']
                                
                                X_test = pd.DataFrame()
                                X_test['lag1']  = data_final.loc[data_final['date']>= cutoff2][ data_final['date']< cutoff3][aggr+'_1'] 
                                X_test['lag2']  = data_final.loc[data_final['date']>= cutoff2][ data_final['date']< cutoff3][aggr+'_2'] 
                                X_test['lag3']  = data_final.loc[data_final['date']>= cutoff2][ data_final['date']< cutoff3][aggr+'_3'] 
                                if lag in ['lags4','lags5','lags6']:
                                    X_test['lag4']  = data_final.loc[data_final['date']>= cutoff2][ data_final['date']< cutoff3][aggr+'_4'] 
                                if lag in ['lags5','lags6']:
                                    X_test['lag5']  = data_final.loc[data_final['date']>= cutoff2][ data_final['date']< cutoff3][aggr+'_5'] 
                                if lag == 'lags6':
                                    X_test['lag6']  = data_final.loc[data_final['date']>= cutoff2][ data_final['date']< cutoff3][aggr+'_6'] 
                                
                                if output == 'abnormal':
                                   y_test = data_final.loc[data_final['date']>= cutoff2][ data_final['date']< cutoff3]['abn_ret']
                                if output == 'intra':
                                    y_test = data_final.loc[data_final['date']>= cutoff2][ data_final['date']< cutoff3]['intra_ret']
                                if output == 'multi':
                                    y_test = data_final.loc[data_final['date']>= cutoff2][ data_final['date']< cutoff3]['intra_multi']
                                    y_dum_reg_test = data_final.loc[data_final['date']>= cutoff2][ data_final['date']< cutoff3]['abn_ret']
                                if output == 'multi2':
                                    y_test = data_final.loc[data_final['date']>= cutoff2][ data_final['date']< cutoff3]['intra_multi2']
                                    y_dum_reg_test = data_final.loc[data_final['date']>= cutoff2][ data_final['date']< cutoff3]['abn_ret']
                                yt_dum = y_test
                                yt_dum[yt_dum<= 0] = 0
                                yt_dum[yt_dum> 0] = 1
                                
                                if output == 'abnormal':
                                   y_test = data_final.loc[data_final['date']>= cutoff2][ data_final['date']< cutoff3]['abn_ret']
                                if output == 'intra':
                                    y_test = data_final.loc[data_final['date']>= cutoff2][ data_final['date']< cutoff3]['intra_ret']
                                if output == 'multi':
                                    yt_dum = data_final.loc[data_final['date']>= cutoff2][ data_final['date']< cutoff3]['intra_multi']
                                if output == 'multi2':
                                    yt_dum = data_final.loc[data_final['date']>= cutoff2][ data_final['date']< cutoff3]['intra_multi2']    
                                
                                #models ----------------------------------------------------------------------------------------
    
                                #ols

                                if '_OLS' in topmodel:
                                    if output != 'multi' and output != 'multi2':
                                        OLS_model = sm.OLS(y, sm.add_constant(X.astype(float))).fit()
                                        y_pred = round((1+np.sign(OLS_model.predict(sm.add_constant(X_test)))).pow(0.0000005),0)
                                        
                                    if output == 'multi2':
                                        OLS_model = sm.OLS(y_dum_reg, sm.add_constant(X.astype(float))).fit()
                                        y_pred2 = OLS_model.predict(sm.add_constant(X_test))
                                        y_pred = pd.DataFrame([2] *(len(y_pred2)))
                                        y_pred.index = y_pred2.index
                                        y_pred[y_pred2<-0.003 ] = 1
                                        y_pred[y_pred2>0.003 ] = 3
                                        y_pred = y_pred.iloc[:,0]
                                    
                                
                                #logistic regression
                                if '_logreg' in topmodel:
                                    log_model = LogisticRegression()

                                    log_model.fit(X,y_dum)  
                                    y_pred = log_model.predict(X_test)
                                                                            
                                #support vector machine
                                if '_SVM' in topmodel:
                                    SVM = svm.SVC(max_iter = 10000)
                                    SVM.fit(X, y_dum)
                                    y_pred = SVM.predict(X_test)
                                    
                                #support vector regression
                                if '_SVR' in topmodel:
                                    if output != 'multi' and output != 'multi2':
                                        SVR_model = svm.SVR(kernel='rbf',max_iter = 10000)
                                        min_max_scaler = preprocessing.MinMaxScaler()
                                        x_scaled = min_max_scaler.fit_transform(X)
                                        min_max_scaler = preprocessing.MinMaxScaler()
                                        y_norm = (y-min(y))/(max(y)-min(y))
                                        x_test_scaled = min_max_scaler.fit_transform(X_test)
                                        SVR_model.fit(x_scaled, y_norm)
                                        y_pred = ((SVR_model.predict(x_test_scaled))*(max(y)-min(y))+min(y))
                                        y_pred[y_pred>0] = 1
                                        y_pred[y_pred<=0] = 0
                                    if output == 'multi2':
                                        SVR_model = svm.SVR(kernel='rbf',max_iter = 10000)
                                        min_max_scaler = preprocessing.MinMaxScaler()
                                        x_scaled = min_max_scaler.fit_transform(X)
                                        min_max_scaler = preprocessing.MinMaxScaler()
                                        y_norm = (y_dum_reg-min(y_dum_reg))/(max(y_dum_reg)-min(y_dum_reg))
                                        x_test_scaled = min_max_scaler.fit_transform(X_test)
                                        SVR_model.fit(x_scaled, y_norm)
                                        y_pred2 = ((SVR_model.predict(x_test_scaled))*(max(y_dum_reg)-min(y_dum_reg))+min(y_dum_reg))
                                        y_pred2 = pd.DataFrame(y_pred2)
                                        y_pred = pd.DataFrame([2]* len(y_pred))
                                        y_pred[y_pred2<-0.003 ] = 1
                                        y_pred[y_pred2>0.003 ] = 3
                                        y_pred = y_pred.iloc[:,0]
                                        y_pred.index = yt_dum.index
                                
                                #Naive Bayes Classifier
                                if '_NB' in topmodel:
                                    if output != 'multi' and output != 'multi2':
                                        bnb = BernoulliNB()
                                        y_pred = bnb.fit(X, y_dum).predict(X_test)
                                    if output == 'multi':
                                        gnb = GaussianNB()
                                        y_pred = gnb.fit(X, y_dum).predict(X_test)
                                    if output == 'multi2':
                                        gnb = GaussianNB()
                                        y_pred = gnb.fit(X, y_dum).predict(X_test)
                                if company == 'HAS':
                                    HAS_res_section2[topmodel] = y_pred
                                if company == 'NVIDIA':
                                    NVIDIA_res_section2[topmodel] = y_pred
                                if company == 'LMT':
                                    LMT_res_section2[topmodel] = y_pred

                                print(str(round(nr/length,2)*100) + " % completed")
                                nr = nr+1
                                
HAS_res_section2.to_csv("HAS_res_section2.csv")
NVIDIA_res_section2.to_csv("NVIDIA_res_section2.csv")
LMT_res_section2.to_csv("LMT_res_section2.csv")

#----------------------------------------------- SECTION 3 ----------------------------------------------------------------------- 
#  The decisions of the different models per company are brought together for the different investment strategies.



HAS_res_section2 = pd.read_csv("HAS_res_section2.csv")
NVIDIA_res_section2= pd.read_csv("NVIDIA_res_section2.csv")
LMT_res_section2 = pd.read_csv("LMT_res_section2.csv")

ret_LMT = pd.read_csv("NVDA.csv")
ret_HAS = pd.read_csv("HAS.csv")
ret_NVDA = pd.read_csv("NVDA.csv")

HAS_res_section2 = HAS_res_section2.replace([np.inf, -np.inf], np.nan).dropna(axis=1).drop(columns="Unnamed: 0")
NVIDIA_res_section2 = NVIDIA_res_section2.replace([np.inf, -np.inf], np.nan).dropna(axis=1).drop(columns="Unnamed: 0")
LMT_res_section2 = LMT_res_section2.replace([np.inf, -np.inf], np.nan).dropna(axis=1).drop(columns="Unnamed: 0")


#for the trading simulation
trading_df_ = pd.DataFrame()
trading_df_['date'] =  ret_LMT.loc[ret_LMT['Date']>= cutoff2][ ret_LMT['Date']< cutoff3][ ret_LMT['Date']< cutoff3]['Date']

trading_df_['HASprice'] = ret_HAS.loc[ret_HAS['Date']>= cutoff2][ ret_HAS['Date']< cutoff3][ ret_HAS['Date']< cutoff3]['Open']
trading_df_['HAS1_day_dif'] = ((ret_HAS['Close'] -  ret_HAS['Open'])/ ret_HAS['Open']).loc[ret_HAS['Date']>= cutoff2][ ret_HAS['Date']< cutoff3]
trading_df_['HASabs_dif_1_day'] = ret_HAS['Close'] -  ret_HAS['Open']

trading_df_['NVDAprice'] = ret_NVDA.loc[ret_NVDA['Date']>= cutoff2][ ret_NVDA['Date']< cutoff3][ ret_NVDA['Date']< cutoff3]['Open']
trading_df_['NVDA1_day_dif'] = ((ret_NVDA['Close'] -  ret_NVDA['Open'])/ ret_LMT['Open']).loc[ret_NVDA['Date']>= cutoff2][ ret_NVDA['Date']< cutoff3]
trading_df_['NVDAabs_dif_1_day'] = ret_NVDA['Close'] -  ret_NVDA['Open']

trading_df_['LMTprice'] = ret_LMT.loc[ret_LMT['Date']>= cutoff2][ ret_LMT['Date']< cutoff3][ ret_LMT['Date']< cutoff3]['Open']
trading_df_['LMT1_day_dif'] = ((ret_LMT['Close'] -  ret_LMT['Open'])/ ret_LMT['Open']).loc[ret_LMT['Date']>= cutoff2][ ret_LMT['Date']< cutoff3]
trading_df_['LMTabs_dif_1_day'] = ret_LMT['Close'] -  ret_LMT['Open']
trading_df_['month'] = trading_df_['date'].str[5:7]

datesvar = trading_df_['date'] 

result_matrix3 = pd.DataFrame()
nr = 0
length = len(HAS_res_section2.columns)*len(NVIDIA_res_section2.columns)*len(LMT_res_section2.columns)
for has in HAS_res_section2.columns:
    for nvidia in NVIDIA_res_section2.columns:
        for lmt in LMT_res_section2.columns:
            budget = 10000
            LMT_res_section3 = LMT_res_section2[lmt]
            if "multi" in lmt:
                pass;
            if "multi2" in lmt:
                LMT_res_section3[LMT_res_section3==3] = 4
                LMT_res_section3[LMT_res_section3==2] = 3
                LMT_res_section3[LMT_res_section3==1] = 2
            if ("multi2" in lmt or "multi" in lmt) == False:
                LMT_res_section3[LMT_res_section3==1] = 4
                LMT_res_section3[LMT_res_section3==0] = 2
            LMT_res_section3.index = datesvar
                
            HAS_res_section3 = HAS_res_section2[has]
            if "multi" in has:
                pass;
            if "multi2" in has:
                HAS_res_section3[HAS_res_section3==3] = 4
                HAS_res_section3[HAS_res_section3==2] = 3
                HAS_res_section3[HAS_res_section3==1] = 2
            if ("multi2" in has or "multi" in has) == False:
                HAS_res_section3[HAS_res_section3==1] = 4
                HAS_res_section3[HAS_res_section3==0] = 2
            HAS_res_section3.index = datesvar
                
            NVIDIA_res_section3 = NVIDIA_res_section2[nvidia]
            if "multi" in nvidia:
                pass;
            if "multi2" in nvidia:
                NVIDIA_res_section3[NVIDIA_res_section3==3] = 4
                NVIDIA_res_section3[NVIDIA_res_section3==2] = 3
                NVIDIA_res_section3[NVIDIA_res_section3==1] = 2
            if ("multi2" in nvidia or "multi" in nvidia) == False:
                NVIDIA_res_section3[NVIDIA_res_section3==1] = 4
                NVIDIA_res_section3[NVIDIA_res_section3==0] = 2
            NVIDIA_res_section3.index = datesvar
            
            port_returns_perc = pd.Series()
            port_returns = pd.Series()
            budgetarray = pd.Series()
            for date in datesvar:
                if LMT_res_section3[date] < 4:
                    lmt_action = 0
                if LMT_res_section3[date] == 4:
                    lmt_action = 1
                if LMT_res_section3[date] == 5:
                    lmt_action = 3
                
                if HAS_res_section3[date] < 4:
                    has_action = 0
                if HAS_res_section3[date] == 4:
                    has_action = 1
                if HAS_res_section3[date] == 5:
                    has_action = 3
                
                if NVIDIA_res_section3[date] < 4:
                    nvidia_action = 0
                if NVIDIA_res_section3[date] == 4:
                    nvidia_action = 1
                if NVIDIA_res_section3[date] == 5:
                    nvidia_action = 3
                
                if has_action == 0 and nvidia_action ==0 and lmt_action == 0:
                    budgetarray[date] = 0
                    port_returns_perc[date] = 0
                    continue;
                    
                #budgetverspreiding
                NVDAprice = float(trading_df_[trading_df_['date']==date]['NVDAprice'])
                LMTprice = float(trading_df_[trading_df_['date']==date]['LMTprice'])
                HASprice = float(trading_df_[trading_df_['date']==date]['HASprice'])
                
                total_stockprices = nvidia_action*NVDAprice +lmt_action*LMTprice +has_action*HASprice
                
                amount = math.floor(budget/total_stockprices)
                rest_budget = budget - (total_stockprices*amount)
                
                has_rest = budget
                nvidia_rest = budget
                lmt_rest = budget
                
                if has_action != 0:
                    has_rest = rest_budget - math.floor(rest_budget/HASprice)*HASprice
                if nvidia_action != 0:
                    nvidia_rest = rest_budget - math.floor(rest_budget/NVDAprice)*NVDAprice
                if lmt_action != 0:
                    lmt_rest = rest_budget - math.floor(rest_budget/LMTprice)*LMTprice
                
                has_ex = 0
                nvidia_ex = 0
                lmt_ex = 0
                
                if has_rest<=nvidia_rest and has_rest<=lmt_rest:
                    has_ex = math.floor(rest_budget/HASprice)
                if nvidia_rest<=has_rest and nvidia_rest<=lmt_rest:
                    nvidia_ex = math.floor(rest_budget/HASprice)
                if lmt_rest<=nvidia_rest and lmt_rest<=has_rest:
                    lmt_ex = math.floor(rest_budget/HASprice)
                
                nvidia_return = float((amount*nvidia_action+nvidia_ex)*trading_df_[trading_df_['date']==date]['NVDAabs_dif_1_day'])
                has_return = float((amount*has_action+has_ex)*trading_df_[trading_df_['date']==date]['HASabs_dif_1_day'])
                lmt_return = float((amount*lmt_action+lmt_ex)*trading_df_[trading_df_['date']==date]['LMTabs_dif_1_day'])
                
                budgetnew = budget + nvidia_return + has_return + lmt_return
                
                port_returns_perc[date] = (budgetnew - budget)/budget
                port_returns[date] = (budgetnew - budget)
                
                budget = budgetnew
                budgetarray[date] = budget
            
            result_matrix3.at[nr,'HAS'] = has
            result_matrix3.at[nr,'NVIDIA'] = nvidia
            result_matrix3.at[nr,'LMT'] = lmt
            result_matrix3.at[nr,'ROR'] = (budgetarray[len(budgetarray)-1] - budgetarray[0])/budgetarray[0]
            result_matrix3.at[nr,'VaR_perc'] = port_returns_perc.sort_values(ascending=True)[int(len(port_returns_perc)/20)-1]
            result_matrix3.at[nr,'VaR'] = port_returns.sort_values(ascending=True)[int(len(port_returns)/20)-1]
            nr = nr+1
            print(str(round((nr/length)*100,2))+" % completed")

result_matrix3.to_csv("res_section3.csv")

#----------------------------------------------- SECTION 4 ----------------------------------------------------------------------- 
# The best model is choosen, based on the VaR and ROR. Which is run together over the 2 year training data en 1 year of test data to get the decisions of the test year.
result_matrix3 = pd.read_csv("res_section3.csv")
themodel = (result_matrix3[result_matrix3['ROR']>=0].sort_values(by='VaR_perc', ascending=False)).iloc[0]   
#themodel = (result_matrix3.sort_values(by='VaR', ascending=False)).iloc[0]    
   



for company in companies:
    topmodel = themodel[company]
    if 'TSV' in topmodel:
        aggregates = ['TSV']
    if 'neg_aggr' in topmodel:
        aggregates = ['neg_aggr']
    if 'neg_aggr2' in topmodel:
        aggregates = ['neg_aggr2']
    if 'sumaggr' in topmodel:
        aggregates = ['sumaggr']
    if 'afinn_score' in topmodel:
        scores = ['afinn_score']
    if 'senti_score' in topmodel:
        scores = ['senti_score']
    if 'OpFi_score' in topmodel:
        scores = ['OpFi_score']
    if 'emoji' in topmodel:
        scores = ['emoji']
    if 'SWN' in topmodel:
        scores = ['SWN']
    if 'intra' in topmodel:
        output_set = ['intra']
    if 'abnormal' in topmodel:
        output_set = ['abnormal']
    if 'multi' in topmodel:
        output_set = ['multi']
    if 'multi2' in topmodel:
        output_set = ['multi2']
    if 'data_full' in topmodel:
        datasets = ['data_full']
    if 'dol_data' in topmodel:
        datasets = ['dol_data']
    if 'data_full_FT' in topmodel:
        datasets = ['data_full_FT']
    if 'dol_data_no_rt' in topmodel:
        datasets = ['dol_data_no_rt']
    if 'data_full_no_rt' in topmodel:
        datasets = ['data_full_no_rt']
    if 'weighted' in topmodel:
        weight_opt = ['weighted']
    if 'non-weighted' in topmodel:
        weight_opt = ['non-weighted']
    if 'lags3' in topmodel:
        lags = ['lags3']
    if 'lags4' in topmodel:
        lags = ['lags4']
    if 'lags5' in topmodel:
        lags = ['lags5']
    if 'lags6' in topmodel:
        lags = ['lags6']
        
    #LMT
    if company == 'LMT':
        if 'data_full' in topmodel:
            data_full = pd.read_csv("LMTdata2016-2018_rev5.csv") 
            data_full['senti_score'] = data_full['senti_score_pos'] + data_full['senti_score_neg']
        if 'dol_data' in topmodel:
            dol_data = pd.read_csv("Dollar_LMT_data2016-2018_rev5.csv")
            dol_data['senti_score'] = dol_data['senti_score_pos'] + dol_data['senti_score_neg']
        if 'data_full_FT' in topmodel:
            data_full_FT = pd.read_csv("LMT_FT_Data2016-2018.csv")
            data_full_FT['senti_score'] = dol_data['senti_score_pos'] + dol_data['senti_score_neg']
        if 'data_full_no_rt' in topmodel:
            data_full = pd.read_csv("LMTdata2016-2018_rev5.csv") 
            data_full['senti_score'] = data_full['senti_score_pos'] + data_full['senti_score_neg']
            data_full_no_rt = data_full.sort_values(by=['date']).drop_duplicates(subset=['text']).sort_values(by=['Unnamed: 0'])
        if 'dol_data_no_rt' in topmodel:
            dol_data = pd.read_csv("Dollar_LMT_data2016-2018_rev5.csv")
            dol_data['senti_score'] = dol_data['senti_score_pos'] + dol_data['senti_score_neg']
            dol_data_no_rt = dol_data.sort_values(by=['date']).drop_duplicates(subset=['text']).sort_values(by=['Unnamed: 0'])
        ret_SP = pd.read_csv("^GSPC.csv") 
        ret_LMT = pd.read_csv("LMT.csv")
        
    #NVIDIA
    if company == 'NVIDIA':
        if 'data_full' in topmodel:
            data_full = pd.read_csv("NVIDIAdata2016-2018_rev5_2.csv") 
            data_full['senti_score'] = data_full['senti_score_pos'] + data_full['senti_score_neg']
        if 'dol_data' in topmodel:
            dol_data = pd.read_csv("dNVIDIAdata2016-2018_rev5_2.csv")
            dol_data['senti_score'] = dol_data['senti_score_pos'] + dol_data['senti_score_neg']
        if 'data_full_FT' in topmodel:
            data_full_FT = pd.read_csv("NVIDIA_FT_Data2016-2018.csv")
            data_full_FT['senti_score'] = dol_data['senti_score_pos'] + dol_data['senti_score_neg']
        if 'data_full_no_rt' in topmodel:
            data_full = pd.read_csv("NVIDIAdata2016-2018_rev5_2.csv") 
            data_full['senti_score'] = data_full['senti_score_pos'] + data_full['senti_score_neg']
            data_full_no_rt = data_full.sort_values(by=['date']).drop_duplicates(subset=['text']).sort_values(by=['Unnamed: 0'])
        if 'dol_data_no_rt' in topmodel:
            dol_data = pd.read_csv("dNVIDIAdata2016-2018_rev5_2.csv")
            dol_data['senti_score'] = dol_data['senti_score_pos'] + dol_data['senti_score_neg']
            dol_data_no_rt = dol_data.sort_values(by=['date']).drop_duplicates(subset=['text']).sort_values(by=['Unnamed: 0'])
        ret_SP = pd.read_csv("^GSPC.csv") 
        ret_LMT = pd.read_csv("NVDA.csv")
    
    #HAS
    if company == 'HAS':
        if 'data_full' in topmodel:
            data_full = pd.read_csv("HASdata2016-2018_rev5_2.csv") 
            data_full['senti_score'] = data_full['senti_score_pos'] + data_full['senti_score_neg']
        if 'dol_data' in topmodel:
            dol_data = pd.read_csv("dHASdata2016-2018_rev5.csv")
            dol_data['senti_score'] = dol_data['senti_score_pos'] + dol_data['senti_score_neg']
        if 'data_full_FT' in topmodel:
            data_full_FT = pd.read_csv("HAS_FT_Data2016-2018.csv")
            data_full_FT['senti_score'] = dol_data['senti_score_pos'] + dol_data['senti_score_neg']
        if 'data_full_no_rt' in topmodel:
            data_full = pd.read_csv("HASdata2016-2018_rev5_2.csv") 
            data_full['senti_score'] = data_full['senti_score_pos'] + data_full['senti_score_neg']
            data_full_no_rt = data_full.sort_values(by=['date']).drop_duplicates(subset=['text']).sort_values(by=['Unnamed: 0'])
        if 'dol_data_no_rt' in topmodel:
            dol_data = pd.read_csv("dHASdata2016-2018_rev5.csv")
            dol_data['senti_score'] = dol_data['senti_score_pos'] + dol_data['senti_score_neg']
            dol_data_no_rt = dol_data.sort_values(by=['date']).drop_duplicates(subset=['text']).sort_values(by=['Unnamed: 0'])
        ret_SP = pd.read_csv("^GSPC.csv") 
        ret_LMT = pd.read_csv("HAS.csv")

        
    #output
    returns = pd.DataFrame()
    returns['date'] = ret_LMT['Date']
    returns['intra_ret'] = ((ret_LMT['Close'] -  ret_LMT['Open'])/ret_LMT['Open'])
    intra_ret_SP = ((ret_SP['Close'] -  ret_SP['Open'])/ret_SP['Open'])
    returns['intra_multi'] = 3
    returns['intra_multi'][returns['intra_ret']<-0.003 ] = 2
    returns['intra_multi'][returns['intra_ret']<-0.02 ] = 1
    returns['intra_multi'][returns['intra_ret']>0.003 ] = 4
    returns['intra_multi'][returns['intra_ret']>0.02 ] = 5
    returns['intra_multi2'] = 2
    returns['intra_multi2'][returns['intra_ret']<-0.003 ] = 1
    returns['intra_multi2'][returns['intra_ret']>0.003 ] = 3
    
    
    
    #abnormal returns
    abn_ret_model = sm.OLS(returns['intra_ret'], sm.add_constant(intra_ret_SP)).fit()
    returns['abn_ret'] = returns['intra_ret'] - abn_ret_model.predict()
    
    #for the trading simulation
    trading_df_ = pd.DataFrame()
    trading_df_['date'] =  ret_LMT.loc[ret_LMT['Date']>= cutoff3][ ret_LMT['Date']< cutoff4]['Date']
    trading_df_['1_day_dif'] = ((ret_LMT['Close'] -  ret_LMT['Open'])/ ret_LMT['Open']).loc[ret_LMT['Date']>= cutoff3][ ret_LMT['Date']< cutoff4]
    trading_df_['abs_dif_1_day'] = ret_LMT['Close'] -  ret_LMT['Open']
    trading_df_['month'] = trading_df_['date'].str[5:7]
    trading_df_BAK = trading_df_
    
    
    #for the trading simulation 2
    trading_df2_ = pd.DataFrame()
    trading_df2_['date'] =  ret_LMT.loc[ret_LMT['Date']>= cutoff3][ ret_LMT['Date']< cutoff4]['Date']
    trading_df2_['1_day_dif'] = ((ret_LMT['Close'] -  ret_LMT['Open'])/ ret_LMT['Open']).loc[ret_LMT['Date']>= cutoff3][ ret_LMT['Date']< cutoff4]
    trading_df2_['abs_dif_1_day'] = ret_LMT['Close'] -  ret_LMT['Open']
    trading_df2_['month'] = trading_df2_['date'].str[5:7]
    trading_df2_BAK = trading_df2_
    
    for dataset in datasets:
        if dataset == 'dol_data':
            data = dol_data
        if dataset == 'data_full':
            data = data_full
        if dataset == 'data_full_no_rt':
            data = data_full_no_rt
        if dataset == 'dol_data_no_rt':
            data = dol_data_no_rt
        if dataset == 'data_full_FT':
            data = data_full_FT
        trading_df = trading_df_ 
        trading_df2 = trading_df2_
        for score in scores:
            model = dataset
            model = model + "_" + score
            # dates = list(set(data['date']))
            if len(dates) != len(data['date']):
                for i in dates:
                    if sum(data['date'] == i) == 0:
                        datalen = (len(data)+1)
                        data.append(pd.Series(name=datalen))
                        for col in data.columns:
                            data.at[datalen,col] = 0
                        data.at[datalen,'date'] = i
            dates.sort()
            pos_count = [0]*len(dates)
            tot_count = [0]*len(dates)
            neg_count = [0]*len(dates)
            TSV_1 = [0]*len(dates)
            sumaggr = [0]*len(dates)
            neg_aggr = pd.DataFrame()
            neg_aggr2 = [0]*len(dates)
            indexer = pd.DataFrame()
            neg_aggr['date'] = dates
            for weight_option in weight_opt:  
                model5 = model + "_" + weight_option
                if weight_option == 'non-weighted':
                    j = 0
                    for i in dates:                        
                        pos_count[j] = len(data[data[score]>0][data['date'] == i].index)
                        tot_count[j] = len(data[data['date'] == i].index)
                        sumaggr[j] = sum(data[score][data['date'] == i])
                        neg_count[j] = tot_count[j] - pos_count[j]   
                        TSV_1[j] = np.log((1+pos_count[j])/(1+neg_count[j]))
                        neg_aggr.at[j,'score'] = neg_count[j]/divide_checker(tot_count[j])
                        if datetime.strptime(i, '%Y-%m-%d').month !=1:
                            month_1 = datetime(datetime.strptime(i, '%Y-%m-%d').year,datetime.strptime(i, '%Y-%m-%d').month-1,1).strftime('%Y-%m-%d')[:7]
                        else:
                            month_1 = datetime(datetime.strptime(i, '%Y-%m-%d').year-1,12,1).strftime('%Y-%m-%d')[:7]
                        neg_aggr.at[j,'TF'] = datetime.strptime(i, '%Y-%m-%d').strftime('%Y-%m-%d')[:7]
                        try:
                            mu_neg = neg_aggr[(neg_aggr['TF'] == month_1)]['score'].mean()
                            sig_neg = np.std(neg_aggr[neg_aggr['TF'] == month_1]['score'])
                            neg_aggr2[j] = (neg_aggr.at[j,'score'] -  mu_neg)/divide_checker(sig_neg)
                        except:
                            neg_aggr2[j] = 0
                        j = j+1
                
                if weight_option == 'weighted':
                    j = 0
                    for i in dates:
                        pos_count[j] = sum(data[data[score]>0][data['date'] == i].weight)
                        tot_count[j] = sum(data[data['date'] == i].weight)
                        neg_count[j] = sum(data[data[score]<0][data['date'] == i].weight) 
                        sumaggr[j] = sum(data[score][data['date'] == i]*data['weight'][data['date'] == i])
                        neg_count[j] = tot_count[j] - pos_count[j]   
                        TSV_1[j] = np.log((1+pos_count[j])/(1+neg_count[j]))
                        neg_aggr.at[j,'score'] = neg_count[j]/divide_checker(tot_count[j])
                        if datetime.strptime(i, '%Y-%m-%d').month !=1:
                            month_1 = datetime(datetime.strptime(i, '%Y-%m-%d').year,datetime.strptime(i, '%Y-%m-%d').month-1,1).strftime('%Y-%m-%d')[:7]
                        else:
                            month_1 = datetime(datetime.strptime(i, '%Y-%m-%d').year-1,12,1).strftime('%Y-%m-%d')[:7]
                        neg_aggr.at[j,'TF'] = datetime.strptime(i, '%Y-%m-%d').strftime('%Y-%m-%d')[:7]
                        try:
                            mu_neg = neg_aggr[(neg_aggr['TF'] == month_1)]['score'].mean()
                            sig_neg = np.std(neg_aggr[neg_aggr['TF'] == month_1]['score'])
                            neg_aggr2[j] = (neg_aggr.at[j,'score'] -  mu_neg)/divide_checker(sig_neg)
                        except:
                            neg_aggr2[j] = 0
                        j = j+1
                neg_aggr2 = [0 if np.isnan(x) else x for x in neg_aggr2]
                #correlation between aggregates
                #corrdia = pd.DataFrame()
                #corrdia['TSV1'] = TSV_1
                #corrdia['TSV2'] = TSV_2
                #corrdia['neg1'] = neg_aggr['score']
                #corrdia['neg2'] = neg_aggr2
                #corrdia.corr()\
                
                
                data2 = pd.DataFrame()
                data2['date'] = dates
                data2['TSV'] = TSV_1
                data2['TSV_1'] = data2['TSV'].shift(periods=1)
                data2['TSV_2'] = data2['TSV'].shift(periods=2)
                data2['TSV_3'] = data2['TSV'].shift(periods=3)
                data2['TSV_4'] = data2['TSV'].shift(periods=4)
                data2['TSV_5'] = data2['TSV'].shift(periods=5)
                data2['TSV_6'] = data2['TSV'].shift(periods=6)
                data2['sumaggr'] = sumaggr
                data2['sumaggr_1'] = data2['sumaggr'].shift(periods=1)
                data2['sumaggr_2'] = data2['sumaggr'].shift(periods=2)
                data2['sumaggr_3'] = data2['sumaggr'].shift(periods=3)
                data2['sumaggr_4'] = data2['sumaggr'].shift(periods=4)
                data2['sumaggr_5'] = data2['sumaggr'].shift(periods=5)
                data2['sumaggr_6'] = data2['sumaggr'].shift(periods=6)
                data2['neg_aggr'] = neg_aggr['score']
                data2['neg_aggr_1'] = data2['neg_aggr'].shift(periods=1)
                data2['neg_aggr_2'] = data2['neg_aggr'].shift(periods=2)
                data2['neg_aggr_3'] = data2['neg_aggr'].shift(periods=3)
                data2['neg_aggr_4'] = data2['neg_aggr'].shift(periods=4)
                data2['neg_aggr_5'] = data2['neg_aggr'].shift(periods=5)
                data2['neg_aggr_6'] = data2['neg_aggr'].shift(periods=6)
                data2['neg_aggr2'] = neg_aggr2
                data2['neg_aggr2_1'] = data2['neg_aggr2'].shift(periods=1)
                data2['neg_aggr2_2'] = data2['neg_aggr2'].shift(periods=2)
                data2['neg_aggr2_3'] = data2['neg_aggr2'].shift(periods=3)
                data2['neg_aggr2_4'] = data2['neg_aggr2'].shift(periods=4)
                data2['neg_aggr2_5'] = data2['neg_aggr2'].shift(periods=5)
                data2['neg_aggr2_6'] = data2['neg_aggr2'].shift(periods=6)
                
                data_final = returns.merge(data2, left_on='date', right_on='date')
                for lag in lags:
                    model6 = model5 + "_" + lag
                    for output in output_set:
                        model2 = model6 + "_" + output
                        for aggr in aggregates:
                            model3 = model2 + "_" + aggr
                            X = pd.DataFrame()
                            X['lag1']  = data_final.loc[data_final['date']< cutoff3][aggr+'_1'] 
                            X['lag2']  = data_final.loc[data_final['date']< cutoff3][aggr+'_2']  
                            X['lag3']  = data_final.loc[data_final['date']< cutoff3][aggr+'_3'] 
                            if lag in ['lags4','lags5','lags6']:
                                X['lag4']  = data_final.loc[data_final['date']< cutoff3][aggr+'_4'] 
                            if lag in ['lags5','lags6']:
                                X['lag5']  = data_final.loc[data_final['date']< cutoff3][aggr+'_5'] 
                            if lag == 'lags6':
                                X['lag6']  = data_final.loc[data_final['date']< cutoff3][aggr+'_6'] 
                            X = X.replace([np.inf, -np.inf], np.nan).dropna()
                            
                            if output == 'abnormal':
                                y = data_final.loc[data_final['date']< cutoff3]['abn_ret']
                            if output == 'intra':
                                y = data_final.loc[data_final['date']< cutoff3]['intra_ret']
                            if output == 'multi':
                                y = data_final.loc[data_final['date']< cutoff3]['intra_multi']
                            if output == 'multi2':
                                y = data_final.loc[data_final['date']< cutoff3]['intra_multi2']    
                                
                            y = y[X.index]
                            y_dum = y
                            y_dum[y_dum<= 0] = 0
                            y_dum[y_dum> 0] = 1
                            
                            if output == 'abnormal':
                                y = data_final.loc[data_final['date']< cutoff3]['abn_ret']
                            if output == 'intra':
                                y = data_final.loc[data_final['date']< cutoff3]['intra_ret']
                            if output == 'multi':
                                y_dum = data_final.loc[data_final['date']< cutoff3]['intra_multi']
                                y_dum = y_dum[X.index]
                                y_dum_reg = data_final.loc[data_final['date']< cutoff3]['abn_ret']
                                y_dum_reg = y_dum_reg[X.index]
                            if output == 'multi2':
                                y_dum = data_final.loc[data_final['date']< cutoff3]['intra_multi2']
                                y_dum = y_dum[X.index]  
                                y_dum_reg = data_final.loc[data_final['date']< cutoff3]['abn_ret']
                                y_dum_reg = y_dum_reg[X.index]
                            y = y[X.index]
                            
                            
                            #y = data_final.loc[data_final['date']< cutoff]['intra_ret']
                            
                            X_test = pd.DataFrame()
                            X_test['lag1']  = data_final.loc[data_final['date']>= cutoff3][ data_final['date']< cutoff4][aggr+'_1'] 
                            X_test['lag2']  = data_final.loc[data_final['date']>= cutoff3][ data_final['date']< cutoff4][aggr+'_2'] 
                            X_test['lag3']  = data_final.loc[data_final['date']>= cutoff3][ data_final['date']< cutoff4][aggr+'_3'] 
                            if lag in ['lags4','lags5','lags6']:
                                X_test['lag4']  = data_final.loc[data_final['date']>= cutoff3][ data_final['date']< cutoff4][aggr+'_4'] 
                            if lag in ['lags5','lags6']:
                                X_test['lag5']  = data_final.loc[data_final['date']>= cutoff3][ data_final['date']< cutoff4][aggr+'_5'] 
                            if lag == 'lags6':
                                X_test['lag6']  = data_final.loc[data_final['date']>= cutoff3][ data_final['date']< cutoff4][aggr+'_6'] 
                            
                            if output == 'abnormal':
                               y_test = data_final.loc[data_final['date']>= cutoff3][ data_final['date']< cutoff4]['abn_ret']
                            if output == 'intra':
                                y_test = data_final.loc[data_final['date']>= cutoff3][ data_final['date']< cutoff4]['intra_ret']
                            if output == 'multi':
                                y_test = data_final.loc[data_final['date']>= cutoff3][ data_final['date']< cutoff4]['intra_multi']
                                y_dum_reg_test = data_final.loc[data_final['date']>= cutoff3][ data_final['date']< cutoff4]['abn_ret']
                            if output == 'multi2':
                                y_test = data_final.loc[data_final['date']>= cutoff3][ data_final['date']< cutoff4]['intra_multi2']
                                y_dum_reg_test = data_final.loc[data_final['date']>= cutoff3][ data_final['date']< cutoff4]['abn_ret']
                            yt_dum = y_test
                            yt_dum[yt_dum<= 0] = 0
                            yt_dum[yt_dum> 0] = 1
                            
                            if output == 'abnormal':
                               y_test = data_final.loc[data_final['date']>= cutoff3][ data_final['date']< cutoff4]['abn_ret']
                            if output == 'intra':
                                y_test = data_final.loc[data_final['date']>= cutoff3][ data_final['date']< cutoff4]['intra_ret']
                            if output == 'multi':
                                yt_dum = data_final.loc[data_final['date']>= cutoff3][ data_final['date']< cutoff4]['intra_multi']
                            if output == 'multi2':
                                yt_dum = data_final.loc[data_final['date']>= cutoff3][ data_final['date']< cutoff4]['intra_multi2']    
                            
                            #models ----------------------------------------------------------------------------------------

                            #ols

                            if '_OLS' in topmodel:
                                if output != 'multi' and output != 'multi2':
                                    OLS_model = sm.OLS(y, sm.add_constant(X.astype(float))).fit()
                                    y_pred = round((1+np.sign(OLS_model.predict(sm.add_constant(X_test)))).pow(0.0000005),0)
                                    
                                if output == 'multi2':
                                    OLS_model = sm.OLS(y_dum_reg, sm.add_constant(X.astype(float))).fit()
                                    y_pred2 = OLS_model.predict(sm.add_constant(X_test))
                                    y_pred = pd.DataFrame([2] *(len(y_pred2)))
                                    y_pred.index = y_pred2.index
                                    y_pred[y_pred2<-0.003 ] = 1
                                    y_pred[y_pred2>0.003 ] = 3
                                    y_pred = y_pred.iloc[:,0]
                                
                            
                            #logistic regression
                            if '_logreg' in topmodel:
                                log_model = LogisticRegression()

                                log_model.fit(X,y_dum)  
                                y_pred = log_model.predict(X_test)
                                                                        
                            #support vector machine
                            if '_SVM' in topmodel:
                                SVM = svm.SVC(max_iter = 10000)
                                SVM.fit(X, y_dum)
                                y_pred = SVM.predict(X_test)
                                
                            #support vector regression
                            if '_SVR' in topmodel:
                                if output != 'multi' and output != 'multi2':
                                    SVR_model = svm.SVR(kernel='rbf',max_iter = 10000)
                                    min_max_scaler = preprocessing.MinMaxScaler()
                                    x_scaled = min_max_scaler.fit_transform(X)
                                    min_max_scaler = preprocessing.MinMaxScaler()
                                    y_norm = (y-min(y))/(max(y)-min(y))
                                    x_test_scaled = min_max_scaler.fit_transform(X_test)
                                    SVR_model.fit(x_scaled, y_norm)
                                    y_pred = ((SVR_model.predict(x_test_scaled))*(max(y)-min(y))+min(y))
                                    y_pred[y_pred>0] = 1
                                    y_pred[y_pred<=0] = 0
                                if output == 'multi2':
                                    SVR_model = svm.SVR(kernel='rbf',max_iter = 10000)
                                    min_max_scaler = preprocessing.MinMaxScaler()
                                    x_scaled = min_max_scaler.fit_transform(X)
                                    min_max_scaler = preprocessing.MinMaxScaler()
                                    y_norm = (y_dum_reg-min(y_dum_reg))/(max(y_dum_reg)-min(y_dum_reg))
                                    x_test_scaled = min_max_scaler.fit_transform(X_test)
                                    SVR_model.fit(x_scaled, y_norm)
                                    y_pred2 = ((SVR_model.predict(x_test_scaled))*(max(y_dum_reg)-min(y_dum_reg))+min(y_dum_reg))
                                    y_pred2 = pd.DataFrame(y_pred2)
                                    y_pred = pd.DataFrame([2]* len(y_pred))
                                    y_pred[y_pred2<-0.003 ] = 1
                                    y_pred[y_pred2>0.003 ] = 3
                                    y_pred = y_pred.iloc[:,0]
                                    y_pred.index = yt_dum.index
                            
                            #Naive Bayes Classifier
                            if '_NB' in topmodel:
                                if output != 'multi' and output != 'multi2':
                                    bnb = BernoulliNB()
                                    y_pred = bnb.fit(X, y_dum).predict(X_test)
                                if output == 'multi':
                                    gnb = GaussianNB()
                                    y_pred = gnb.fit(X, y_dum).predict(X_test)
                                if output == 'multi2':
                                    gnb = GaussianNB()
                                    y_pred = gnb.fit(X, y_dum).predict(X_test)
                            if company == 'HAS':
                                HAS_themodel = y_pred
                            if company == 'NVIDIA':
                                NVIDIA_themodel = y_pred
                            if company == 'LMT':
                                LMT_themodel = y_pred
    
    
    
#----------------------------------------------- FINAL SECTION --------------------------------------------------------------------------------------------------------------
# The decisions of the models are brought together in one investment strategy and tested.


ret_LMT = pd.read_csv("NVDA.csv")
ret_HAS = pd.read_csv("HAS.csv")
ret_NVDA = pd.read_csv("NVDA.csv")



#for the trading simulation
trading_df_ = pd.DataFrame()
trading_df_['date'] =  ret_LMT.loc[ret_LMT['Date']>= cutoff3][ ret_LMT['Date']< cutoff4][ ret_LMT['Date']< cutoff4]['Date']

trading_df_['HASprice'] = ret_HAS.loc[ret_HAS['Date']>= cutoff3][ ret_HAS['Date']< cutoff4][ ret_HAS['Date']< cutoff4]['Open']
trading_df_['HAS1_day_dif'] = ((ret_HAS['Close'] -  ret_HAS['Open'])/ ret_HAS['Open']).loc[ret_HAS['Date']>= cutoff3][ ret_HAS['Date']< cutoff4]
trading_df_['HASabs_dif_1_day'] = ret_HAS['Close'] -  ret_HAS['Open']

trading_df_['NVDAprice'] = ret_NVDA.loc[ret_NVDA['Date']>= cutoff3][ ret_NVDA['Date']< cutoff4][ ret_NVDA['Date']< cutoff4]['Open']
trading_df_['NVDA1_day_dif'] = ((ret_NVDA['Close'] -  ret_NVDA['Open'])/ ret_LMT['Open']).loc[ret_NVDA['Date']>= cutoff3][ ret_NVDA['Date']< cutoff4]
trading_df_['NVDAabs_dif_1_day'] = ret_NVDA['Close'] -  ret_NVDA['Open']

trading_df_['LMTprice'] = ret_LMT.loc[ret_LMT['Date']>= cutoff3][ ret_LMT['Date']< cutoff4][ ret_LMT['Date']< cutoff4]['Open']
trading_df_['LMT1_day_dif'] = ((ret_LMT['Close'] -  ret_LMT['Open'])/ ret_LMT['Open']).loc[ret_LMT['Date']>= cutoff3][ ret_LMT['Date']< cutoff4]
trading_df_['LMTabs_dif_1_day'] = ret_LMT['Close'] -  ret_LMT['Open']
trading_df_['month'] = trading_df_['date'].str[5:7]

datesvar = trading_df_['date'] 

HAS_themodel2 = pd.DataFrame(data=HAS_themodel,columns=[themodel['HAS']])
NVIDIA_themodel2 = pd.DataFrame(data=NVIDIA_themodel,columns=[themodel['NVIDIA']])
LMT_themodel2 = pd.DataFrame(data=LMT_themodel,columns=[themodel['LMT']])


result_matrix4 = pd.DataFrame()
nr = 0
length = len(HAS_res_section2.columns)*len(NVIDIA_res_section2.columns)*len(LMT_res_section2.columns)
for has in HAS_themodel2.columns:
    for nvidia in NVIDIA_themodel2.columns:
        for lmt in LMT_themodel2.columns:
            budget = 10000
            LMT_res_section3 = LMT_themodel2[lmt]
            if "multi" in lmt:
                pass;
            if "multi2" in lmt:
                LMT_res_section3[LMT_res_section3==3] = 4
                LMT_res_section3[LMT_res_section3==2] = 3
                LMT_res_section3[LMT_res_section3==1] = 2
            if ("multi2" in lmt or "multi" in lmt) == False:
                LMT_res_section3[LMT_res_section3==1] = 4
                LMT_res_section3[LMT_res_section3==0] = 2
            LMT_res_section3.index = datesvar
                
            HAS_res_section3 = HAS_themodel2[has]
            if "multi" in has:
                pass;
            if "multi2" in has:
                HAS_res_section3[HAS_res_section3==3] = 4
                HAS_res_section3[HAS_res_section3==2] = 3
                HAS_res_section3[HAS_res_section3==1] = 2
            if ("multi2" in has or "multi" in has) == False:
                HAS_res_section3[HAS_res_section3==1] = 4
                HAS_res_section3[HAS_res_section3==0] = 2
            HAS_res_section3.index = datesvar
                
            NVIDIA_res_section3 = NVIDIA_themodel2[nvidia]
            if "multi" in nvidia:
                pass;
            if "multi2" in nvidia:
                NVIDIA_res_section3[NVIDIA_res_section3==3] = 4
                NVIDIA_res_section3[NVIDIA_res_section3==2] = 3
                NVIDIA_res_section3[NVIDIA_res_section3==1] = 2
            if ("multi2" in nvidia or "multi" in nvidia) == False:
                NVIDIA_res_section3[NVIDIA_res_section3==1] = 4
                NVIDIA_res_section3[NVIDIA_res_section3==0] = 2
            NVIDIA_res_section3.index = datesvar
            
            port_returns_perc = pd.Series()
            port_returns = pd.Series()
            budgetarray = pd.Series()
            for date in datesvar:
                if LMT_res_section3[date] < 4:
                    lmt_action = 0
                if LMT_res_section3[date] == 4:
                    lmt_action = 1
                if LMT_res_section3[date] == 5:
                    lmt_action = 3
                
                if HAS_res_section3[date] < 4:
                    has_action = 0
                if HAS_res_section3[date] == 4:
                    has_action = 1
                if HAS_res_section3[date] == 5:
                    has_action = 3
                
                if NVIDIA_res_section3[date] < 4:
                    nvidia_action = 0
                if NVIDIA_res_section3[date] == 4:
                    nvidia_action = 1
                if NVIDIA_res_section3[date] == 5:
                    nvidia_action = 3
                
                if has_action == 0 and nvidia_action ==0 and lmt_action == 0:
                    budgetarray[date] = budget
                    port_returns_perc[date] = 0
                    continue;
                    
                #budgetverspreiding
                NVDAprice = float(trading_df_[trading_df_['date']==date]['NVDAprice'])
                LMTprice = float(trading_df_[trading_df_['date']==date]['LMTprice'])
                HASprice = float(trading_df_[trading_df_['date']==date]['HASprice'])
                
                total_stockprices = nvidia_action*NVDAprice +lmt_action*LMTprice +has_action*HASprice
                
                amount = math.floor(budget/total_stockprices)
                rest_budget = budget - (total_stockprices*amount)
                
                has_rest = budget
                nvidia_rest = budget
                lmt_rest = budget
                
                if has_action != 0:
                    has_rest = rest_budget - math.floor(rest_budget/HASprice)*HASprice
                if nvidia_action != 0:
                    nvidia_rest = rest_budget - math.floor(rest_budget/NVDAprice)*NVDAprice
                if lmt_action != 0:
                    lmt_rest = rest_budget - math.floor(rest_budget/LMTprice)*LMTprice
                
                has_ex = 0
                nvidia_ex = 0
                lmt_ex = 0
                
                if has_rest<=nvidia_rest and has_rest<=lmt_rest:
                    has_ex = math.floor(rest_budget/HASprice)
                if nvidia_rest<=has_rest and nvidia_rest<=lmt_rest:
                    nvidia_ex = math.floor(rest_budget/HASprice)
                if lmt_rest<=nvidia_rest and lmt_rest<=has_rest:
                    lmt_ex = math.floor(rest_budget/HASprice)
                
                nvidia_return = float((amount*nvidia_action+nvidia_ex)*trading_df_[trading_df_['date']==date]['NVDAabs_dif_1_day'])
                has_return = float((amount*has_action+has_ex)*trading_df_[trading_df_['date']==date]['HASabs_dif_1_day'])
                lmt_return = float((amount*lmt_action+lmt_ex)*trading_df_[trading_df_['date']==date]['LMTabs_dif_1_day'])
                
                budgetnew = budget + nvidia_return + has_return + lmt_return
                
                port_returns_perc[date] = (budgetnew - budget)/budget
                port_returns[date] = (budgetnew - budget)
                
                budget = budgetnew
                budgetarray[date] = budget
            
            result_matrix4.at[nr,'HAS'] = has
            result_matrix4.at[nr,'NVIDIA'] = nvidia
            result_matrix4.at[nr,'LMT'] = lmt
            result_matrix4.at[nr,'ROR'] = (budgetarray[len(budgetarray)-1] - 10000)/10000
            result_matrix4.at[nr,'VaR_perc'] = port_returns_perc.sort_values(ascending=True)[int(len(port_returns_perc)/20)-1]
            result_matrix4.at[nr,'VaR'] = port_returns.sort_values(ascending=True)[int(len(port_returns)/20)-1]


result_matrix4.to_csv("res_section42.csv")

result_matrix4 = pd.read_csv("res_section4.csv")

#PLOTS
ret_SPplot = pd.read_csv("^GSPC2.csv") 
ret_SPplot.index = ret_SPplot['Date']
SP500ret = ret_SPplot.loc[budgetarray.index]['Close']*(10000/ret_SPplot.loc[budgetarray.index]['Close'][0])
plt.style.use('seaborn-whitegrid')
fig = plt.figure()
ax = plt.axes()
ax.plot(budgetarray,label='Portfolio');
ax.plot(SP500ret,label='S&P500');
ax.legend()
ax.set(xlabel='Time (2018)', ylabel='Budget (USD)',
       title='Portfolio vs S&P500 performance')
ax.axes.get_xaxis().set_ticks([])
ax.grid()

fig.savefig("themodel_budget.png")

ret_LMTplot = pd.read_csv("LMT.csv") 
ret_LMTplot.index = ret_LMTplot['Date']
LMTret = ret_LMTplot.loc[budgetarray.index]['Close']*(10000/ret_LMTplot.loc[budgetarray.index]['Close'][0])
ret_HASplot = pd.read_csv("HAS.csv") 
ret_HASplot.index = ret_HASplot['Date']
HASret = ret_HASplot.loc[budgetarray.index]['Close']*(10000/ret_HASplot.loc[budgetarray.index]['Close'][0])
ret_NVDAplot = pd.read_csv("NVDA.csv") 
ret_NVDAplot.index = ret_NVDAplot['Date']
NVDAret = ret_NVDAplot.loc[budgetarray.index]['Close']*(10000/ret_NVDAplot.loc[budgetarray.index]['Close'][0])
plt.style.use('seaborn-whitegrid')
fig = plt.figure()
ax = plt.axes()
ax.plot(budgetarray,label='Portfolio');
ax.plot(LMTret,label='LMT');
ax.plot(HASret,label='HAS');
ax.plot(NVDAret,label='NVDA');
ax.legend()
ax.set(xlabel='Time (2018)', ylabel='Budget (USD)',
       title='Portfolio vs individual stock performance')
ax.axes.get_xaxis().set_ticks([])
ax.grid()

fig.savefig("themodel_v_stocks_budget.png")


SP500_change = (SP500ret - SP500ret.shift(periods=1))/SP500ret.shift(periods=1)
port_change = (budgetarray - budgetarray.shift(periods=1))/budgetarray.shift(periods=1)

plt.style.use('seaborn-whitegrid')
fig = plt.figure()
ax = plt.axes()
ax.plot(port_change,label='Portfolio');
ax.plot(SP500_change,label='S&P500');
ax.legend()
ax.set(xlabel='Time (2018)', ylabel='Budget change (%)',
       title='Portfolio vs S&P500 day-to-day change')
ax.axes.get_xaxis().set_ticks([])
ax.grid()

fig.savefig("themodel_budget_change.png")

Sharpe_ratio = (port_change.mean() - (1.01**(1/365)-1))/port_change.std() 
SP500ret_sharpe = (SP500_change.mean() - (1.01**(1/365)-1))/SP500_change.std() 

LMTret.std()
HASret.std()
NVDAret.std()
budgetarray.std()
SP500ret.std()

(SP500ret[len(SP500ret)-1]-10000)/10000

(LMTret - LMTret.shift(periods=1)).sort_values(ascending=True)[int(len(LMTret)/20)-1]
(HASret - HASret.shift(periods=1)).sort_values(ascending=True)[int(len(HASret)/20)-1]
(NVDAret - NVDAret.shift(periods=1)).sort_values(ascending=True)[int(len(NVDAret)/20)-1]
(SP500ret - SP500ret.shift(periods=1)).sort_values(ascending=True)[int(len(SP500ret)/20)-1]
port_returns.sort_values(ascending=True)[int(len(port_returns)/20)-1]
