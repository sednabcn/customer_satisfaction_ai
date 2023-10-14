# Code taken from Machine Learning Mastery
# Author: Jason Brownlee
# example of loading and preparing the dataset
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
#from sklearn.preprocessing import OrdinalEncoder
from category_encoders import OrdinalEncoder 

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# load the dataset
def load_dataset(filename):
 # load the dataset as a pandas DataFrame
 data = pd.read_csv(filename,dtype='str').astype('category')
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
 fs = SelectKBest(score_func=mutual_info_classif, k=k_f)
 fs.fit(X_train, y_train)
 nfeatures_out=fs.get_feature_names_out(input_features=features_in)
 #print(nfeatures_out)
 X_train_fs = fs.transform(X_train)
 X_test_fs = fs.transform(X_test)
 return X_train_fs, X_test_fs

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
