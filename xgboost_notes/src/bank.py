import os
from urllib.request import urlretrieve
from zipfile import ZipFile
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import xgboost as xgb

NUM_TRAIN = 40000
DATA_FOLDER = Path('../data/')
ZIP_FILE = '../data/bank.zip'
DATA_FILE = DATA_FOLDER/'bank-full.csv'

# Download dataset if not already present
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip'

if not os.path.isfile(ZIP_FILE):
    urlretrieve(url, ZIP_FILE)
    with ZipFile(ZIP_FILE, 'r') as zf:
        zf.extractall()

bank = pd.read_csv(DATA_FILE, sep=';')

is_object = bank.apply(lambda x: x.dtype == 'object')
categorical_vars = is_object.index[is_object].tolist()

ctr = ColumnTransformer([
    ('encode', OrdinalEncoder(), categorical_vars),
], remainder='passthrough')

ctr.fit(bank)
transformed = ctr.transform(bank)
y = transformed[:, 9]
X = np.delete(transformed, 9, axis=1)

x_test = X[NUM_TRAIN:]
y_test = y[NUM_TRAIN:]

X = X[:NUM_TRAIN]
y = y[:NUM_TRAIN]

x_train, x_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y)

dtrain = xgb.DMatrix(x_train, label=y_train)
dval = xgb.DMatrix(x_val, label=y_val)
dtest = xgb.DMatrix(x_test, label=y_test)

w = np.round((len(y) - y_train.sum()) / y_train.sum())
evallist = [(dval, 'eval'), (dtrain, 'train')]
param = {'max_depth': 2, 'eta': 1, 'objective': 'binary:logistic',
         'scale_pos_weight': 10}
bst = xgb.train(param, dtrain, 10, evallist, early_stopping_rounds=3)
probs = bst.predict(dval, ntree_limit=bst.best_ntree_limit)
ypreds = (probs > 0.5).astype(np.int)
print(classification_report(y_val, ypreds))
