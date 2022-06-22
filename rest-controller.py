'''
Created on Apr 3, 2022

@author: Avadhut Shelar
'''
import json
from flask import Flask, request
import pandas as pd
import py_eureka_client.eureka_client as eureka_client
import pickle
from pip._vendor.rich.jupyter import display

rest_port = 8050
eureka_client.init(eureka_server="http://localhost:8761/eureka",
                   app_name="smartlaw-ml-service",
                   instance_port=rest_port)

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, SmartLawML!</p>"


@app.route("/predictArgumentBy", methods=['POST'])
def predictArgumentBy():
    data = request.json
    
    argumentTexts=data["argumentTexts"]
        
    loaded_model_vect = pickle.load(open("E:/git/SmartLawML/models/model_vect_argby.pkl", 'rb'))    
    loaded_model_argby = pickle.load(open("E:/git/SmartLawML/models/best_model_argby.pkl", 'rb'))
    
    argBy=[]
    
    for argText in argumentTexts:    
        X_vec=loaded_model_vect.transform([argText])
        temp = loaded_model_argby.predict(X_vec.toarray())[0]
        argBy.append(temp)
        
    x = {
      "argumentBy":argBy
    }
    
    # convert into JSON:
    response = json.dumps(x)
    
    return response

@app.route("/predictSentenceType", methods=['POST'])
def predictSentenceType():
    data = request.json
    
    sentenceTypeTexts=data["sentenceTypeTexts"]
        
    loaded_model_vect = pickle.load(open("E:/git/SmartLawML/models/model_vect_senttype.pkl", 'rb'))    
    loaded_model_senttype = pickle.load(open("E:/git/SmartLawML/models/best_model_senttype.pkl", 'rb'))
    
    sentType=[]
    
    for sentenceTypeText in sentenceTypeTexts:    
        X_vec=loaded_model_vect.transform([sentenceTypeText])
        temp = loaded_model_senttype.predict(X_vec.toarray())[0]
        sentType.append(temp)
        
    x = {
      "sentenceType":sentType
    }
    
    # convert into JSON:
    response = json.dumps(x)
    
    return response

@app.route("/predictOrderType", methods=['POST'])
def predictOrderType():
    data = request.json
    
    orderTypeTexts=data["orderTypeTexts"]
        
    loaded_model_vect = pickle.load(open("E:/git/SmartLawML/models/model_vect_ordertype.pkl", 'rb'))    
    loaded_model_ordertype = pickle.load(open("E:/git/SmartLawML/models/best_model_ordertype.pkl", 'rb'))
    
    oType=[]
    
    for orderTypeText in orderTypeTexts:    
        X_vec=loaded_model_vect.transform([orderTypeText])
        temp = loaded_model_ordertype.predict(X_vec.toarray())[0]
        oType.append(temp)
        
    x = {
      "orderType":oType
    }
    
    # convert into JSON:
    response = json.dumps(x)
    
    return response

@app.route("/getDatasetStats", methods=['GET'])
def getDatasetStats():
    import sys
    sys.path.append('models')
    from datasetStats import getDatasetStatistics
    out = getDatasetStatistics()
   
    x = {
      "datasetStatisticsList":out
    }
    response = json.dumps(x)
    return response

@app.route("/getCVResults", methods=['GET'])
def getCVResults():
    import sys
    sys.path.append('models')
    from sentencetype import cvResult
    sentenceTypeCVResult = cvResult()
    from argumentby import cvResult
    argumentbyCVResult = cvResult()
    from ordertype import cvResult
    ordertypeCVResult = cvResult()
    x = {
      "sentenceTypeCVResult":sentenceTypeCVResult,
      "argumentbyCVResult":argumentbyCVResult,
      "ordertypeCVResult":ordertypeCVResult
    }
    
    json_object = json.dumps(x, indent = 4)
    import time
    timestr = time.strftime("%Y%m%d-%H%M%S")  
    filename="temp/CVResults"+timestr+".json"  
    # Writing to sample.json
    with open(filename, "w") as outfile:
        outfile.write(json_object)
    
    response = json.dumps(x)
    return response

@app.route("/submitModelMetricsJob", methods=['POST'])
def submitModelMetricsJob():
    import sys
    sys.path.append('models')
    from sentencetype import modelMetrics
    sentenceTypeModelMetrics = modelMetrics()
    from argumentby import modelMetrics
    argumentbyModelMetrics = modelMetrics()
    from ordertype import modelMetrics
    ordertypeModelMetrics = modelMetrics()
    x = {
      "sentenceTypeModelMetrics":sentenceTypeModelMetrics,
      "argumentbyModelMetrics":argumentbyModelMetrics,
      "ordertypeModelMetrics":ordertypeModelMetrics
    }
    
    json_object = json.dumps(x, indent = 4)
    import time
    timestr = time.strftime("%Y%m%d-%H%M%S")  
    filename="temp/ModelMetrics"+timestr+".json"
    
    
    # Writing to sample.json
    with open(filename, "w") as outfile:
        outfile.write(json_object)

    
    response = json.dumps(x)
    return response


@app.route("/getModelMetrics", methods=['GET'])
def getModelMetrics():
    import sys
    sys.path.append('models')
        
    filename="temp/ModelMetrics.json"  
    with open(filename,encoding="utf8") as f1: 
        response=json.load(f1)
        
    return response
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port = rest_port)