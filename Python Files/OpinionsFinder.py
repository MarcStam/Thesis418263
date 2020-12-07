import pandas as pd
import os
import math
import numpy as np
def OpFi(name,n,data):
    os.chdir("C:/Users/marcs/OneDrive/Bureaublad/Master/Thesis")
    
    # data = pd.read_csv("LMT_data2016-2018.csv") 
    # name = "LMT"
    
    # data = pd.read_csv("NVIDIA_data.csv") 
    # data4 = pd.read_csv("NVIDIA_data2017_2.csv") 
    # data3 =pd.read_csv("NVIDIA_data2016_2.csv") 
    # data = data.append(data4)
    # data = data.append(data3)
    # data = data.drop_duplicates()  
    data = pd.read_csv("dHASdata_2016-2018.csv")
    name = "dHAS"
    data2 = data['text']
    data2 = data2.tolist()
    length = len(data2)
    
    os.chdir("C:/Users/marcs/OneDrive/Bureaublad/Master/Thesis/opinionfinderv2.0/database/docs/"+name)
    n = 5000
    
   
    
    for i in list(range(len(data2))):
        f= open(str(i),"w+")
        f.write(str(data2[i].encode('utf-8')))
    

    os.chdir("C:/Users/marcs/OneDrive/Bureaublad/Master/Thesis/opinionfinderv2.0")
    
    iter = math.ceil(length/n)
    for j in list(range(iter)):
        string = ""
        if j<(iter-1):
            for k in list(range(j*n,(j+1)*n)):
                string = string+ "database/docs/"+name+"/"+str(k)+"\n"
        else:
            for k in list(range(j*n,len(data2))):
                string = string+"database/docs/"+name+"/"+str(k)+"\n"
        file = open(name+ str(j)+".doclist","w+")
        file.write(string)
        file.close()
        os.system('java -Xmx1g -classpath lib\weka.jar;lib\stanford-postagger.jar;opinionfinder.jar opin.main.RunOpinionFinder '+name+str(j)+'.doclist -d')
        
    
    OpFi_result = pd.DataFrame()
    os.chdir("C:/Users/marcs/OneDrive/Bureaublad/Master/Thesis/opinionfinderv2.0/database/docs/"+name)
    for i in list(range(length)):    
        try:
            file= open(str(i)+"_auto_anns/exp_polarity.txt","r")
            lines = file.readlines()
            #analyze polarity
            countPolarity = 0
            count_number = 0
            for l in lines:
               	if "negative" in l:
               	    countPolarity = countPolarity - 1;  count_number += 1
               	elif "positive" in l:
               		countPolarity = countPolarity + 1
               		count_number += 1
               	elif  "neutral" in l:
               		countNeutral = countPolarity + 0
               	elif "weakpos" in l:
               		countPolarity = countPolarity + 1
               		count_number += 1
               	elif "strongpos" in l:
               		countPolarity = countPolarity + 2
               		count_number += 1
            OpFi_result.at[i,0] = countPolarity
            OpFi_result.at[i,1] = count_number
        except:
            pass
        if i % 1000 == 0:
            print("At: "+str(i) +" of "+str(length))
    
    data['OpFi_score'] = OpFi_result[0]
    data['OpFi_weight'] =  OpFi_result[1]
    
    os.chdir("C:/Users/marcs/OneDrive/Bureaublad/Master/Thesis")
    data.to_csv(name+"data2016-2018_rev.csv")
    return(data)

