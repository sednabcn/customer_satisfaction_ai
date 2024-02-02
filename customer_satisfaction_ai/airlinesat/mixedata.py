import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from category_encoders import OrdinalEncoder 
import numpy as np
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

#========================
# load the dataset
def load_dataset(filename):
 # load the dataset as a pandas DataFrame
 df = pd.read_csv(filename,index_col=0)
 df = df.drop(df.iloc[:,[0]],axis=1)
 
 # for only one column
 clm=list(df.columns[df.isna().sum()>0])
 value=df[clm].median()
 df.replace(np.nan,value,inplace=True)
 
 cat_cols=[col for col in df.columns if df[col].dtype=="object"]
 ord_cols= [col for col in df.columns if min(df[col]) in {0,1} and max(df[col])==5]
 cont_cols= [col for col in df.columns if col not in ord_cols + cat_cols]
 columns=df.columns
 # retrieve numpy array
 dfset = df.values
 # split into input (X) and output (y) variables
 X = dfset[:, :-1]
 y = dfset[:,-1]
 # format all fields as string
 #X = X.astype(str)
 return X, y,columns,cat_cols[:-1],ord_cols,cont_cols

# load the dataset

def transf_dataset(X,cols,cat_cols,ord_cols,cont_cols):
    """ transformation using embedding to numerical datasets"""
    df_comb_cols=[]
    X=pd.DataFrame(X,columns=cols[:-1])
    # categorical
    df_cat=X[cat_cols]
    df_cat_enc = pd.get_dummies(df_cat,columns=cat_cols,dtype='int')
    df_comb_cols.extend(df_cat_enc.columns)
    # ordinal
    df_ord=X[ord_cols]
    ord_mapping = lambda x:x
    df_ord_map = df_ord.applymap(ord_mapping)
    df_ord_enc =pd.get_dummies(df_ord_map,columns=ord_cols, dtype='int')
    df_comb_cols.extend(df_ord_enc.columns)
    # continuous
    df_cont=X[cont_cols]
    df_cont_scaled = scaler.fit_transform(df_cont)
    df_comb_cols.extend(df_cont.columns)
    # combined
    df_comb = np.hstack((df_cat_enc,df_ord_enc,df_cont_scaled)) 
    return df_comb, df_comb_cols


X, y,feature_names_in,nominal_cols,ordinal_cols, numerical_cols = load_dataset('Airlinessat.csv')

X,Xcols=transf_dataset(X,feature_names_in,nominal_cols,ordinal_cols, numerical_cols)

# make a new X

# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=1)


X_traind =pd.DataFrame(X_train,columns=Xcols)
X_testd = pd.DataFrame(X_test,columns=Xcols)

assert X_traind.shape[0] + X_testd.shape[0]==X.shape[0]
print(X_traind.head(3))
print(X_traind.shape[1])
print(X_test.shape[1])
"""
──(agagora㉿kali)-[~/Downloads/CONSUEXP/airlinesat/PYTHON_CODES]
└─$ python mixedata.py
   Gender_Female  Gender_Male  ...  Departure Delay in Minutes  Arrival Delay in Minutes
0            0.0          1.0  ...                    1.793657                  1.378443
1            0.0          1.0  ...                   -0.386481                 -0.391644
2            1.0          0.0  ...                    0.611655                  0.363246

[3 rows x 96 columns]
96
96
"""
