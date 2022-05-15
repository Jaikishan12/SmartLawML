import json
import glob
import pandas as pd
import string
import re

def readInitial():
    #LOCATE EXACT LOCATION OF ALL JSON FILES IN KAGGLE LOCAL SETUP
    train_files = glob.glob("E:/git/SmartLawDataset/json/*.json")
    
    #EXTRACT INDIVIDUAL FILES AND GET THEM INTO SINGLE DATAFRAME
    df_final=pd.DataFrame()
    for file in train_files:
        with open(file,encoding="utf8") as f1: 
            data=json.load(f1) #ALL DATA FROM FILE LOADED INTO DATA
            df=pd.DataFrame([data]) #CREATE DATAFRAME FOR SINGLE DATA
            df_final=pd.concat([df_final,df]) #MERGE ALL INDIVIDUAL DATAFRAME INTO SINGLE DATAFRAME
    #MAIN AIM IS TO GENERATE ML ALGO TO DETERMINE ARGUMENT TEXT AND ARGUMENT BY SO REMOVE REST
    return df_final
    
def preprocess(df_final1):
    s =  set(string.punctuation)
    s.add('\xad')
    for index, row in df_final1.iterrows():
        for x in row['text']:
            row['text']=row['text'].lower()
    #         print(row['text'])
            if x in s or re.search(r'-?\d+', x): 
                row['text']=row['text'].replace(x,"")     
    return df_final1
           
def getArgByDataSet():
    df_final=readInitial()
    
    df_final=df_final.drop(['header','background','order','footer','annotationProcessingStage','annotationProcessingStageAnnotations','processedText'],axis=1)
    #AXIS=1 REMOVE COLUMNS
    
    data1=[]
    data2=[]
    df_final1=pd.DataFrame();
    for j in df_final['arguments']:
        for k in j:
            s=k['text'][3:]
            data1.append(s.strip())
            data2.append(k['argumentBy'])
    df_final1['text']=data1
    df_final1['argumentBy']=data2
    #Use df_final1 dataframe only for further processes and not csv files to ensure encoding is in proper format
    
    df_final1=preprocess(df_final1)    
                
    #print(df_final1)
    return df_final1

def getSentenceTypeDataSet():
    df_final=readInitial()
    
    data1=[]
    data2=[]
    df_final1=pd.DataFrame();
    for i in df_final['arguments']:
        for j in i:
            for k in j['argumentSentences']:          
                s=k['text']
                data1.append(s.strip())
                data2.append(k['argumentSentenceType'])
    df_final1['text']=data1
    df_final1['argumentSentenceType']=data2
    #Use df_final1 dataframe only for further processes and not csv files to ensure encoding is in proper format
    
    df_final1=preprocess(df_final1)    
                
    return df_final1
    
def getOrderTypeDataSet():
    df_final=readInitial()
    
    data1=[]
    data2=[]
    df_final1=pd.DataFrame();
    for i in df_final['order']:        
        s=i['text']
        data1.append(s.strip())
        data2.append(i['orderType'])
    df_final1['text']=data1
    df_final1['orderType']=data2
    #Use df_final1 dataframe only for further processes and not csv files to ensure encoding is in proper format
    
    df_final1=preprocess(df_final1)    
                
    return df_final1