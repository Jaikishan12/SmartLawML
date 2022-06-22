import pandas as pd
import sklearn as sk
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize

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
from sklearn.model_selection import StratifiedKFold
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
    models.append(LogisticRegression()) #
    models.append(RidgeClassifier())
    models.append(SGDClassifier())
    models.append(PassiveAggressiveClassifier())
    models.append(KNeighborsClassifier()) #
    models.append(DecisionTreeClassifier()) #
    models.append(LinearSVC())
    models.append(SVC()) #
    models.append(GaussianNB())
    models.append(AdaBoostClassifier())
    models.append(BaggingClassifier())
    models.append(RandomForestClassifier()) #
    models.append(ExtraTreesClassifier()) 
    #models.append(GaussianProcessClassifier())
    models.append(GradientBoostingClassifier()) 
    models.append(LinearDiscriminantAnalysis())
    models.append(QuadraticDiscriminantAnalysis())
    return models

def evaluate_model_LOOCV(X, y, model): 
    # evaluate the model
    cv = LeaveOneOut()
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    print('LOOCV accuracy=%.3f (%.3f,%.3f)' % (mean(scores), scores.min(), scores.max()))
    # return scores    
    return float("{:.4f}".format(mean(scores))), float("{:.4f}".format(scores.min())), float("{:.4f}".format(scores.max()))

# evaluate the model using a given test condition
def evaluate_model_CV(X, y, num_folds, model):        
    # evaluate the model
    cv = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=1)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    # return scores
    return float("{:.4f}".format(mean(scores))), float("{:.4f}".format(scores.min())), float("{:.4f}".format(scores.max()))

def getCVResults(Xoriginal,yoriginal):
    # get the list of models to consider
    models = get_models()
    # evaluate each model
    X_size=Xoriginal.shape[0]
    n_samples_list = list()
    #n_samples_list.append(int(X_size*0.1))
    n_samples_list.append(int(X_size*0.25))
    #n_samples_list.append(int(X_size*0.3))
    #n_samples_list.append(int(X_size*0.4))
    #n_samples_list.append(int(X_size*0.5))
    #n_samples_list.append(int(X_size*1.0))    
    
    out = list()
    
    iter = 1
    for n_samples in n_samples_list:        
        # collect results
        all_model_out = list()
        ideal_results_all_models, cv_results_all_models = list(), list()
        # get the dataset
        X, y = get_dataset(Xoriginal,yoriginal,n_samples)
        print("(X,y) shape",X.shape,y.shape)
        for model in models: 
            print("Evaluating Model:",type(model).__name__)
            #Evaluate Ideal case
            ideal_mean,ideal_min,ideal_max = evaluate_model_LOOCV(X,y, model)            
            #Evaluate cross validation
            cv_results_current_model = list()
            current_model_cv_out = list()
            num_folds = range(2,11) 
            for k in num_folds:
                cv_mean,cv_min,cv_max = evaluate_model_CV(X,y,k,model)
                # store results
                cv_results_current_model.append(cv_mean)                
                print('> fold=%d, accuracy=%.3f (%.3f,%.3f)' % (k, cv_mean,cv_min,cv_max))
                current_model_cv_out.append({'fold':k,'meanAccuracy':cv_mean,'minAccuracy':cv_min,'maxAccuracy':cv_max})
              
            # check for invalid results
            if isnan(mean(cv_results_current_model)) or isnan(ideal_mean):
                continue
            
            ideal_results_all_models.append(ideal_mean)
            cv_results_all_models.append(mean(cv_results_current_model))
            # summarize progress
            #print('>%s: ideal=%.3f, cv=%.3f' % (type(model).__name__, ideal_mean, cv_mean))
            all_model_out.append({'modelName':type(model).__name__,'meanLOOCV':ideal_mean,'mean10FoldCV':float("{:.4f}".format(mean(cv_results_current_model))),"foldWiseResult":current_model_cv_out})
        #print('Mean LOOCV =%.3f, Mean 10-fold CV =%.3f' % (mean(ideal_results),mean(cv_results)))   
        out.append({'iterationNumber':iter,'noOfSamples':n_samples,'meanLOOCVAllModels':float("{:.4f}".format(mean(ideal_results_all_models))),'mean10FoldCVAllModels':float("{:.4f}".format(mean(cv_results_all_models))),'mlModelResultList':all_model_out})
        #out.append({'iterationNumber':iter,'noOfSamples':n_samples,'meanLOOCVAllModels':mean(ideal_results),'mean10FoldCVAllModels':mean(cv_results)})
        iter = iter + 1 
    
    return out


def getModelMetrics(Xoriginal,yoriginal):
    # get the list of models to consider
    models = get_models()
    
    out = list()
       
    # collect results
    all_model_out = list()
    p_r_f1_all_models = list()
    # get the dataset
    X, y = Xoriginal,yoriginal
    class_labels=y.unique()
    y = label_binarize(y, classes=class_labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=0)
    print("(X,y) shape",X.shape,y.shape)
    for model in models: 
        print("Evaluating Model:",type(model).__name__)
        
        classifier = OneVsRestClassifier(model)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        results = {}               
        results['confusion_matrix'] = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)).tolist()
        print ("\n\nConfusion Matrix:\n",confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)))
        
        results['classification_report'] = classification_report(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1),digits=4,target_names=class_labels,output_dict=True)
        model_report = classification_report(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1),digits=4,target_names=class_labels)
        print(model_report)

        out.append({'modelName':type(model).__name__,'results':results})

    return out

def getModelMetricsOrderType(Xoriginal,yoriginal):
    # get the list of models to consider
    models = get_models()
    
    out = list()
       
    # collect results
    all_model_out = list()
    p_r_f1_all_models = list()
    # get the dataset
    X, y = Xoriginal,yoriginal
    class_labels=y.unique()
    y = label_binarize(y, classes=class_labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=0)
    print("(X,y) shape",X.shape,y.shape)
    for model in models: 
        print("Evaluating Model:",type(model).__name__)
        
        classifier = OneVsRestClassifier(model)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        results = {}               
        results['confusion_matrix'] = confusion_matrix(y_test,y_pred).tolist()
        print ("\n\nConfusion Matrix:\n",confusion_matrix(y_test,y_pred))
        
        results['classification_report'] = classification_report(y_test,y_pred,digits=4,target_names=class_labels,output_dict=True)
        model_report = classification_report(y_test,y_pred,digits=4,target_names=class_labels)
        print(model_report)

        out.append({'modelName':type(model).__name__,'results':results})

    return out