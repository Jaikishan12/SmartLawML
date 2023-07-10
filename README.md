# SmartLawML

In this project we compared 4 to 5 machine learning and deep learning models to be able to solve two related problem statements based on text - classification and receive a high accuracy percentage for the same. We performed required pre-processing on the models, split the dataset into train-test and trained the models. Various accuracy measures are calculated and are tabulated further in the report. The overall analysis gave accuracy of over 99% for prediction of argument by using Deep Learning model (BI-LSTM). For prediction of the class order type, higher accuracies of over 90% were obtained from basic Machine Learning models like SVM (Support Vector Machine).

#### Preparing the local python virtual environment and activating it
Run the following commands - Windows
- cd E:\git\SmartLawML 
- python -m venv smartlawml-microservice

Activate the environment
- smartlawml-microservice\Scripts\activate

#### Installing the required packages for ML project
On the virtual environemnt activated above, install the necessary packages
- (smartlawml-microservice) E:\git\SmartLawML>pip install flask pandas py-eureka-client
- (smartlawml-microservice) E:\git\SmartLawML>pip install jupyter keras tensorflow
- (smartlawml-microservice) E:\git\SmartLawML>pip install nltk sklearn wheel wordcloud
- (smartlawml-microservice) E:\git\SmartLawML>python -m nltk.downloader words
- (smartlawml-microservice) E:\git\SmartLawML>python -m nltk.downloader stopwords
- (smartlawml-microservice) E:\git\SmartLawML>python -m nltk.downloader punkt averaged_perceptron_tagger

#### Starting the ML project as eureka server microservice
On the virtual environemnt activated above, start the ml service
-  (smartlawml-microservice) E:\git\SmartLawML>python rest-controller.py
