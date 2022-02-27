import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree
import warnings
warnings.filterwarnings('ignore')

PATH = '/content/Crop_recommendation.csv'
df = pd.read_csv(PATH)

features = df[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
target = df['label']
labels = df['label']

# Initializing empty lists to append all model's name and corresponding name
acc = []
model = []

# Splitting into train and test data

from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(features,target,test_size = 0.2,random_state =2)


from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(n_estimators=20, random_state=0)
RF.fit(Xtrain,Ytrain)

predicted_values = RF.predict(Xtest)

x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('RF')

import pickle
# Dump the trained Naive Bayes classifier with Pickle
RF_pkl_filename = 'RandomForest.pkl'
# Open the file to save as pkl file
RF_Model_pkl = open(RF_pkl_filename, 'wb')
pickle.dump(RF, RF_Model_pkl)
# Close the pickle instances
RF_Model_pkl.close()

# N	P	K	temperature	humidity	ph	rainfall	
a1 = float(input('Enter Nitrogen content in soil:'))
a2 = float(input('Enter Phosphorus content in soil:'))
a3 = float(input('Enter Potassium content in soil:'))
a4 = float(input('Enter Temperature:'))
a5 = float(input('Enter Humidity:'))
a6 = float(input('Enter pH of soil:'))
a7 = float(input('Enter Rainfall(mm):'))
data = np.array([[a1,a2, a3, a4, a5, a6, a7]])
prediction = LogReg.predict(data)
print(prediction)