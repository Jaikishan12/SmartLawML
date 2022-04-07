# SmartLawML

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