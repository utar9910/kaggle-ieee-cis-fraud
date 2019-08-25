import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

# not using id for now.
# train_ids = pd.read_csv('train_identity.csv')
# test_ids = pd.read_csv('test_identity.csv')


train_trans = pd.read_csv('train_transaction.csv')
test_trans = pd.read_csv('test_transaction.csv')
train_trans = reduce_mem_usage(train_trans)
test_trans = reduce_mem_usage(test_trans)
train_x = train_trans.drop(
    ['TransactionID', 'TransactionDT', 'isFraud'], axis=1)
train_y = train_trans['isFraud'].values
test_x = test_trans.drop(['TransactionID', 'TransactionDT'], axis=1)

# Converts categorical features into ints
for f in train_x.columns:
    if train_x[f].dtype=='object' or test_x[f].dtype=='object': 
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train_x[f].values) + list(test_x[f].values))
        train_x[f] = lbl.transform(list(train_x[f].values))
        test_x[f] = lbl.transform(list(test_x[f].values))   

x_train, x_valid, y_train, y_valid = train_test_split(
    train_x, train_y, test_size=0.1, random_state=42)

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)
watchlist = [(d_train, 'train'), (d_valid, 'valid')]

print('Training ...')
max_steps = 10000
params = {}
max_depth = 1
params['eta'] = 0.02
params['objective'] = 'reg:logistic'
params['eval_metric'] = 'rmse'
params['max_depth'] = 1
params['silent'] = 1

clf = xgb.train(params, d_train, max_steps, watchlist,
                early_stopping_rounds=100, verbose_eval=10)

clf.save_model('xgboost_0825.model')