import pickle
import warnings
import string
import re
warnings.filterwarnings("ignore")

def pre_process(text):
    s =  set(string.punctuation)
    text=text.lower()
    for x in text:
        if x in s or re.search(r'-?\d+', x): 
            text=text.replace(x,"")
    return text

def predict_arg_by(text):
    text=pre_process(text)
    with open("model_vect","rb") as f:
        loaded_model_vect = pickle.load(f)
    X_vec=loaded_model_vect.transform([text])
    X_vec=X_vec.toarray()
    with open("model_arg_by","rb") as f:
        loaded_model_argby = pickle.load(f)
    return ''.join(loaded_model_argby.predict(X_vec))

output1=predict_arg_by("Heard learned advocate Mr. Waghmare for the applicant")
print(output1)
output2=predict_arg_by("Prosecution case is that on 28/12/2018 at about 12.30 a.m. at Narpatgiri chowk, in front of Royal Motors, Mangalwar peth, Pune, applicant/accused on count of somebody threw stone in front of his shop, shouted and abused, therefore, police staff tried to catch him, he abused, beat to the police persons by kicks and fist,and also threatened them.")
print(output2)
output3=predict_arg_by("Learned counsel for the applicant/accused submitted that applicant/accused has not committed any offence, he has been falsely implicated in this offence. He further submitted that there was no any motive or intention on the part of applicant/accused to commit offence. He further submitted that there is no recovery or discovery at the hands of applicant/accused, therefore, applicant be released on bail.")
print(output3)