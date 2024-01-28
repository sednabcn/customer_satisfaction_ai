from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from category_encoders import OrdinalEncoder 
import numpy as np
from sklearn.metrics import accuracy_score
from matplotlib import pyplot

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

# split into train and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=1)

# prepare input data

X_train_enc, X_test_enc = prepare_inputs(X_train, X_test)

# prepare output data

y_train_enc, y_test_enc = prepare_targets(y_train, y_test)


#print("==================================================")

# Choose a classification algorithm (Random Forest as an example)
model = RandomForestClassifier()
model.fit(X_train_enc, y_train_enc)

# Make predictions on the test set
y_pred = model.predict(X_test_enc)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test_enc, y_pred))
print("Classification Report:\n", classification_report(y_test_enc, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test_enc, y_pred))
"""
#===============================RESULTS=================================
┌──(agagora㉿kali)-[~/Downloads/CONSUEXP/airlinesat/PYTHON_CODES]
└─$ python mixedclass01.py
['id', 'Gender', 'Customer Type', 'Age', 'Type of Travel', 'Class', 'Flight Distance', 'Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking', 'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort', 'Inflight entertainment', 'On-board service', 'Leg room service', 'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']
Test size:0.33
Accuracy: 0.9540141387275145
Classification Report:

               precision    recall  f1-score   support
               

               
           0       0.95      0.97      0.96     24381
           
           1       0.96      0.93      0.95     18480
           

           
    accuracy                           0.95     42861
    
   macro avg       0.96      0.95      0.95     42861
   
weighted avg       0.95      0.95      0.95     42861



Confusion Matrix:

 [[23723   658]
  
 [ 1313 17167]]

Test size:0.10
Accuracy: 0.9563443178318448
Classification Report:
               precision    recall  f1-score   support

           0       0.95      0.98      0.96      7396
           1       0.97      0.93      0.95      5592

    accuracy                           0.96     12988
   macro avg       0.96      0.95      0.96     12988
weighted avg       0.96      0.96      0.96     12988

Confusion Matrix:
 [[7218  178]
 [ 389 5203]]

"""
