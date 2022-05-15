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