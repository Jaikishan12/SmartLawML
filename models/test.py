import json
#from datasetStats import getDatasetStatistics
#response = json.dumps(getDatasetStatistics())
#print(response)
from rest-controller import getCVResults
response = json.dumps(getCVResults())
print(response)