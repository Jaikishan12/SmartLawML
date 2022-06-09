import pandas as pd
import sklearn as sk

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


def getBestModel(X_train, X_test, y_train, y_test):
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    #y_pred_train = gnb.predict(X_train)
    print('Model accuracy score using TF_IDF- GNB for predicting argument by: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
    
    best_accuracy = accuracy_score(y_test, y_pred)
    best_model = GaussianNB() 
    
    from sklearn.svm import SVC
    svclassifier = SVC(kernel='linear')
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)
    print('Model accuracy score using TF_IDF- SVC for predicing argument by: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
    
    if ((accuracy_score(y_test, y_pred)>best_accuracy)):
        best_accuracy=accuracy_score(y_test, y_pred)
        best_model = SVC(kernel='linear')
    
    from sklearn.neighbors import KNeighborsClassifier  
    classifier= KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )  
    classifier.fit(X_train, y_train)
    y_pred= classifier.predict(X_test)
    print('Model accuracy score using TF_IDF- KNN for predicting argument by: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
    
    if ((accuracy_score(y_test, y_pred)>best_accuracy)):
        best_accuracy=accuracy_score(y_test, y_pred)
        best_model = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )
        
    #print(best_accuracy)
    
    return best_model


# correlation between test harness and ideal test condition
from numpy import mean
from numpy import isnan
from numpy import asarray
from numpy import polyfit
from scipy.stats import pearsonr
from matplotlib import pyplot
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# create the dataset
def get_dataset(X,y,n_samples=100):
    #X, y = make_classification(n_samples=n_samples, n_features=20, n_informative=15, n_redundant=5, random_state=1)
    return X.sample(n_samples), y.sample(n_samples)

# get a list of models to evaluate
def get_models():
    models = list()
    models.append(LogisticRegression())
    models.append(RidgeClassifier())
    models.append(SGDClassifier())
    models.append(PassiveAggressiveClassifier())
    models.append(KNeighborsClassifier())
    models.append(DecisionTreeClassifier())
    models.append(ExtraTreeClassifier())
    models.append(LinearSVC())
    models.append(SVC())
    models.append(GaussianNB())
    models.append(AdaBoostClassifier())
    models.append(BaggingClassifier())
    models.append(RandomForestClassifier())
    models.append(ExtraTreesClassifier())
    models.append(GaussianProcessClassifier())
    models.append(GradientBoostingClassifier())
    #models.append(LinearDiscriminantAnalysis())
    #models.append(QuadraticDiscriminantAnalysis())
    return models

# evaluate the model using a given test condition
def evaluate_model(X, y, cv, model,n_samples):
    # get the dataset
    X, y = get_dataset(X,y,n_samples)
    # evaluate the model
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    # return scores
    return mean(scores)

def getBestModelCV(X,y):
    # define test conditions
    ideal_cv = LeaveOneOut()
    cv = KFold(n_splits=10, shuffle=True, random_state=1)
    # get the list of models to consider
    models = get_models()
    # evaluate each model
    
    n_samples_list = list()
    n_samples_list.append(350)
    #n_samples_list.append(400)
    
    
    out = list()
    
    iter = 1
    for n_samples in n_samples_list:        
        # collect results
        temp_out = list()
        ideal_results, cv_results = list(), list()
        for model in models:
            # evaluate model using each test condition
            cv_mean = evaluate_model(X,y,cv, model,n_samples)
            ideal_mean = evaluate_model(X,y,ideal_cv, model,n_samples)
            # check for invalid results
            if isnan(cv_mean) or isnan(ideal_mean):
                continue
            # store results
            cv_results.append(cv_mean)
            ideal_results.append(ideal_mean)
            # summarize progress
            #print('>%s: ideal=%.3f, cv=%.3f' % (type(model).__name__, ideal_mean, cv_mean))
            temp_out.append({'modelName':type(model).__name__,'meanLOOCV':ideal_mean,'mean10FoldCV':cv_mean})
        #print('Mean LOOCV =%.3f, Mean 10-fold CV =%.3f' % (mean(ideal_results),mean(cv_results)))   
        out.append({'iterationNumber':iter,'noOfSamples':n_samples,'meanLOOCVAllModels':mean(ideal_results),'mean10FoldCVAllModels':mean(cv_results),'mlModelResultList':temp_out})
        #out.append({'iterationNumber':iter,'noOfSamples':n_samples,'meanLOOCVAllModels':mean(ideal_results),'mean10FoldCVAllModels':mean(cv_results)})
        iter = iter + 1 
    
    return out