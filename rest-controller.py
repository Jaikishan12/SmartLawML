'''
Created on Apr 3, 2022

@author: Avadhut Shelar
'''
import json
from flask import Flask, request
import pandas as pd
import py_eureka_client.eureka_client as eureka_client
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
def hello():
    data = request.json
    
    o1 = data["input1"] + "o1"
    o2 = data["input2"] + "o2"
    
    x = {
      "output1": o1,
      "output2": o2
    }
    
    # convert into JSON:
    response = json.dumps(x)
    
    return response

if __name__ == "__main__":
    app.run(host='0.0.0.0', port = rest_port)