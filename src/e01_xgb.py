"""
    This file runs xgboost, train with train data and predict test data.
"""

import pandas as pd
import numpy as np
import gc
import xgboost as xgb
from sklearn.externals import joblib
from sklearn.cross_validation import StratifiedKFold
from util_logger import get_logger

import sys
argvs = sys.argv 
_ , runtype, version = argvs
LOG = get_logger()
LOG.info("start e01")

def run_xgboost(labels, weights, data):

    # convert data into xgb.DMatrix. 
    # train using 80% of the data, 20% of the data is used for watchlist
    skf = StratifiedKFold(labels, 5, random_state = 123)
    idx_train, idx_test = list(skf)[0]

    dtrain = xgb.DMatrix(data[idx_train,:], weight=weights[idx_train], label=labels[idx_train], missing=np.nan)
    dvalid = xgb.DMatrix(data[idx_test,:], weight=weights[idx_test], label=labels[idx_test], missing=np.nan)
    watchlist = [(dtrain, 'train'),(dvalid, 'eval')]

    # set parameters
    # logistic regression with gradient boosted decision trees
    num_rounds = 1000
    params={'objective': 'reg:logistic',
            'eta': 0.05,
            'max_depth': 6,
            'subsample': 0.85,
            'colsample_bytree': 0.8,
            "silent" : 1,
            "seed" : 12345,
            "min_child_weight" : 1
            }

    # run xgboost
    LOG.info("started xgb")
    model = xgb.train(params, dtrain, num_boost_round=num_rounds, evals=watchlist, early_stopping_rounds=5)

    return model

def predict_and_write(in_fname, out_fname):
    xgbdata = joblib.load(in_fname) 

    # to reduce memory consumption, predict by small chunks.
    def predict(data):
        dtest = xgb.DMatrix(data, missing=np.nan)
        return pd.Series(model.predict(dtest)) 

    data_list = np.array_split(xgbdata["data"], 8)
    preds = pd.concat([predict(data) for data in data_list])
    preds.to_csv(out_fname)

# train model
xgbdata_train = joblib.load( '../model/xgbdata_train.pkl') 
model = run_xgboost(xgbdata_train["labels"], xgbdata_train["weights"], xgbdata_train["data"])
model.save_model('../model/xgb.model')

# release memory
xgbdata_train = None
gc.collect()

# predict test data and write
predict_and_write( '../model/xgbdata_test.pkl', "../model/xgb_predicted.txt") 

LOG.info("finished")

