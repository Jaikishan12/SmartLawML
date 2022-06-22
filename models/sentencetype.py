import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def retrain():
    from smartlawdata import getSentenceTypeDataSet
    df_final1 = getSentenceTypeDataSet()
    #print(df_final1)
    
    los=[]
    for item in df_final1['text']:
        los.append(item)
    
    #Create a TFIDF vectorizer to generate text entered into vector form to be given as input to Machine Learning model
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(los)
    feature_names = vectorizer.get_feature_names_out() #Extract the feature names as columns for the texts
    dense = vectors.todense()
    denselist = dense.tolist()
    df_end = pd.DataFrame(denselist, columns=feature_names)
    df_end['argumentSentenceType']=df_final1['argumentSentenceType']
    
    y=df_end.argumentSentenceType
    x=df_end[feature_names]
    # Setting up x and y coordinates
    #print(x.head())
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)
    # check the shape of X_train and X_test
    #X_train.shape, X_test.shape
    
    from commonmodels import getBestModel
    best_model=getBestModel(X_train, X_test, y_train, y_test);
    
    best_model.fit(x.values,y)
    
    # Creation of pickle file for best model to predict argument by
    model_file = "best_model_senttype.pkl"
    with open(model_file,'wb') as f:
        pickle.dump(best_model, f)
        
    # Creation of pickle file for vectorizer to get the text into vector form for prediction using Machine Learning
    model_file="model_vect_senttype.pkl"
    with open(model_file,'wb') as f:
        pickle.dump(vectorizer, f)
    
def predict():
    loaded_model_vect = pickle.load(open("E:/git/SmartLawML/models/model_vect_senttype.pkl", 'rb'))
    X_vec=loaded_model_vect.transform(["heard learned advocate mr waghmare for the applicant"])
    
    loaded_model_argby = pickle.load(open("E:/git/SmartLawML/models/best_model_senttype.pkl", 'rb'))
    o1 = loaded_model_argby.predict(X_vec.toarray())[0]
    print(o1)
    
def cvResult():
    from smartlawdata import getSentenceTypeDataSet
    df_final1 = getSentenceTypeDataSet()
    #print(df_final1)
    
    los=[]
    for item in df_final1['text']:
        los.append(item)
    
    #Create a TFIDF vectorizer to generate text entered into vector form to be given as input to Machine Learning model
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(los)
    feature_names = vectorizer.get_feature_names_out() #Extract the feature names as columns for the texts
    dense = vectors.todense()
    denselist = dense.tolist()
    df_end = pd.DataFrame(denselist, columns=feature_names)
    df_end['argumentSentenceType']=df_final1['argumentSentenceType']
    
    y=df_end.argumentSentenceType
    X=df_end[feature_names]
    print("(X,y) shape",X.shape,y.shape)
    from commonmodels import getCVResults
    out = getCVResults(X,y)
    #print(out)
    return out

def modelMetrics():
    from smartlawdata import getSentenceTypeDataSet
    df_final1 = getSentenceTypeDataSet()
    #print(df_final1)
    
    los=[]
    for item in df_final1['text']:
        los.append(item)
    
    #Create a TFIDF vectorizer to generate text entered into vector form to be given as input to Machine Learning model
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(los)
    feature_names = vectorizer.get_feature_names_out() #Extract the feature names as columns for the texts
    dense = vectors.todense()
    denselist = dense.tolist()
    df_end = pd.DataFrame(denselist, columns=feature_names)
    df_end['argumentSentenceType']=df_final1['argumentSentenceType']
    
    y=df_end.argumentSentenceType
    X=df_end[feature_names]
    print("(X,y) shape",X.shape,y.shape)
    from commonmodels import getModelMetrics
    out = getModelMetrics(X,y)
    #print(out)
    return out