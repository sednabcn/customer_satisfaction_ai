# Code taken from Machine Learning Mastery
# Author: Jason Brownlee
# example of loading and preparing the dataset
# modify for testing the best_features selection based on ranking
# working with category_encoders rather than sklearn Ordinal Encoder
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
#from sklearn.preprocessing import OrdinalEncoder
from category_encoders import OrdinalEncoder 

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif,chi2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
#========================
feature_selection=2
k_f=18
#========================
# load the dataset
def load_dataset(filename):
 # load the dataset as a pandas DataFrame
 data = pd.read_csv(filename,index_col=0,dtype='str').astype('category')
 columns=data.columns[:-1]
 
 # retrieve numpy array
 dataset = data.values
 # split into input (X) and output (y) variables
 X = dataset[:, :-1]
 y = dataset[:,-1]
 # format all fields as string
 X = X.astype(str)
 return X, y,columns
  
# feature selection
def select_features(X_train, y_train, X_test,k_f,features_in):
 fs = SelectKBest(score_func=chi2, k=k_f)
 fs = SelectKBest(score_func=mutual_info_classif, k=k_f)
 fs.fit(X_train, y_train)
 nfeatures_out=fs.get_feature_names_out(input_features=features_in)
 #print(nfeatures_out)
 X_train_fs = fs.transform(X_train)
 X_test_fs = fs.transform(X_test)
 return X_train_fs, X_test_fs,fs

# prepare input data
def prepare_inputs(X_train, X_test):
 oe = OrdinalEncoder()
 pp=oe.fit(X_train)
 X_train_enc = oe.transform(X_train)
 X_test_enc = oe.transform(X_test)
 return X_train_enc, X_test_enc
 
# prepare target
def prepare_targets(y_train, y_test):
 le = LabelEncoder()
 le.fit(y_train)
 y_train_enc = le.transform(y_train)
 y_test_enc = le.transform(y_test)
 return y_train_enc, y_test_enc
 
# load the dataset
X, y,feature_names_in = load_dataset('airlinessat.csv')
print(list(feature_names_in))
#print(input("PAUSE"))
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# prepare input data
X_train_enc, X_test_enc = prepare_inputs(X_train, X_test)
# prepare output data
y_train_enc, y_test_enc = prepare_targets(y_train, y_test)
#print(X_train_enc.shape)
#print(X_train_enc)
#print("==================================================")
#print(X_test_enc)
# feature selection
if feature_selection==1:
   k_f_max=X_train_enc.shape[1]
   for k_f in range(2,k_f_max):
       X_train_fs, X_test_fs = select_features(X_train_enc, y_train_enc, X_test_enc,k_f,feature_names_in)
       # fit the model
       model = LogisticRegression(solver='lbfgs',max_iter=2000)
       model.fit(X_train_fs, y_train_enc)
       # evaluate the model
       yhat = model.predict(X_test_fs)
       # evaluate predictions
       accuracy = accuracy_score(y_test_enc, yhat)*100
       print('#Features: %2d Accuracy: %.2f' % (k_f,accuracy))
elif feature_selection==2:
       index_col=[4,5,9,12,13,14,15,16,17,18,19,20]
       k_f=len(index_col)
       X_train_fs=X_train_enc[index_col]
       X_test_fs=X_test_enc[index_col]
       # fit the model
       model = LogisticRegression(solver='lbfgs',max_iter=2000)
       model.fit(X_train_fs, y_train_enc)
       # evaluate the model
       yhat = model.predict(X_test_fs)
       # evaluate predictions
       accuracy = accuracy_score(y_test_enc, yhat)*100
       print('#Features: %2d Accuracy: %.2f' % (k_f,accuracy))
else:
       
       X_train_fs, X_test_fs,fs = select_features(X_train_enc, y_train_enc, X_test_enc,k_f,feature_names_in)
       # fit the model
       model = LogisticRegression(solver='lbfgs',max_iter=2000)
       model.fit(X_train_fs, y_train_enc)
       # evaluate the model
       yhat = model.predict(X_test_fs)
       # evaluate predictions
       accuracy = accuracy_score(y_test_enc, yhat)*100
       print('#Features: %2d Accuracy: %.2f' % (k_f,accuracy))
       # what are scores for the features
       for i in range(len(fs.scores_)):
        print('Feature %d: %f' % (i, fs.scores_[i]*100))
        # plot the scores
       pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_*100)
       pyplot.show()

#==========================TEST==============================================
#──(agagora㉿kali)-[~/Downloads/CONSUEXP/airlinesat]
#└─$ python oechi2.py
#['id', 'Gender', 'Customer Type', 'Age', 'Type of Travel', 'Class', 'Flight Distance', 'Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking', 'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort', 'Inflight entertainment', 'On-board service', 'Leg room service', 'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']
#Features:  2 Accuracy: 75.71
#Features:  3 Accuracy: 78.48
#Features:  4 Accuracy: 76.10
#Features:  5 Accuracy: 75.63
#Features:  6 Accuracy: 75.77
#Features:  7 Accuracy: 76.68
#Features:  8 Accuracy: 78.14
#Features:  9 Accuracy: 77.88
#Features: 10 Accuracy: 77.90
#Features: 11 Accuracy: 78.37
#Features: 12 Accuracy: 78.15
#Features: 13 Accuracy: 78.25
#Features: 14 Accuracy: 80.34
#Features: 15 Accuracy: 80.44
#Features: 16 Accuracy: 80.51
#Features: 17 Accuracy: 80.55
#Features: 18 Accuracy: 80.65
#Features: 19 Accuracy: 74.98
#Features: 20 Accuracy: 81.07
#Feature_selction: fscores_
"""
Feature 0: 0.668430
Feature 1: 0.595547
Feature 2: 2.521433
Feature 3: 0.842207
Feature 4: 12.105810
Feature 5: 13.623597
Feature 6: 2.503905
Feature 7: 6.303905
Feature 8: 0.376502
Feature 9: 5.147830
Feature 10: 0.750261
Feature 11: 0.914042
Feature 12: 9.056819
Feature 13: 5.141034
Feature 14: 9.201309
Feature 15: 5.971402
Feature 16: 6.338000
Feature 17: 4.487073
Feature 18: 3.409953
Feature 19: 4.323109
Feature 20: 3.197040
Feature 21: 0.139712
Feature 22: 0.127744
"""
# Selction_of_features_contribution greater than 1
k_f=12 Poor performance
#Features: 12 Accuracy: 76.68
