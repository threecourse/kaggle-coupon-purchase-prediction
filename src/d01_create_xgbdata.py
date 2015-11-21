"""
    This file creates input data for xgboost.
"""

import pandas as pd
import numpy as np
import gc
import multiprocessing as mp
from sklearn import preprocessing
from sklearn.externals import joblib
from util import Utility, Process
from util_logger import get_logger
from dxx_dataframe_preprocess import create_dataframes_with_features
LOG = get_logger()
LOG.info("start d01")

def create_input_and_write(is_train):

    if is_train:
        train_ids = pd.read_pickle("../model/train_ids.pkl")
        xgbdata = create_input_multiprocess(train_ids)
        joblib.dump(xgbdata, '../model/xgbdata_train.pkl') 
        LOG.info("finished train write")
    else:
        test_ids = pd.read_pickle("../model/test_ids.pkl")
        xgbdata = create_input_multiprocess(test_ids)
        joblib.dump(xgbdata, '../model/xgbdata_test.pkl') 
        LOG.info("finished test write")

def create_input_multiprocess(ids):
    """Create input in multi process."""
    
    threads = 8
    p = mp.Pool(threads)
    pool_results = p.map(create_input, np.array_split(ids, threads * 2))   
    p.close()
    p.join()

    xgbdfs = pd.concat(pool_results, ignore_index=True) 

    # release memory
    pool_results = None
    gc.collect()

    xgbdata = to_xgbdata(xgbdfs)

    # release memory
    xgbdfs = None
    gc.collect()

    return xgbdata

def create_input(ids):
    df = ids.copy()
    df = create_dataframes_with_features(df)
    xgbdfs = to_xgb_df(df)

    # release memory
    df = None   
    gc.collect()
    return xgbdfs

def to_xgb_df(_df):
    """Convert dataframe into xgb values dataframe."""

    LOG.info("start to_xgb_df")

    pref_to_id = pd.read_pickle("../model/pref_to_id.pkl")
    le_genre = preprocessing.LabelEncoder().fit(Utility.genres)
 
    df = pd.DataFrame(None, index=_df.index)

    df["y"] = np.where(_df["y"] == 1, 1, 0)
    df["count"] = _df["count"]

    df["genre"] = le_genre.transform(_df.genre) # should consider
    df["user_pref"] = pref_to_id.reindex(_df.user_pref).values 
    df["pref0"]  = np.where(_df.pref0, 1, 0) 
    df["pref24"] = np.where(_df.pref24, 1, 0) 
    df["pref"]  =  pref_to_id.reindex(_df.pref).values 
    df["lnpop"] = _df["lnpop"]
    df["d2"] = _df["d2"]
    df["d2bin"] = _df["d2bin"]
    df["area"] = _df["area"]
    df["SEX_ID"] = np.where(_df["SEX_ID"] == 'f', 1, 0)
    df["AGE"] = _df["AGE"]
    df["spot"] = np.where(_df.spot, 1, 0) 
    df["user_prefNN"] = np.where(_df.user_prefNN, 1,0) 
    df["pb_same_sarea"] = _df["pb_same_sarea"]
    df["pb_same_v_sarea"] = _df["pb_same_v_sarea"]

    df = pd.concat([df, _df[Utility.genres]], axis=1)
    df = pd.concat([df, _df[Utility.prefs]], axis=1)
    df = pd.concat([df, _df[Utility.v_genres]], axis=1)
    df = pd.concat([df, _df[Utility.v_prefs]], axis=1)
     
    df["pb_same_genre"] = _df["pb_same_genre"]
    df["pb_same_v_genre"] = _df["pb_same_v_genre"]
    df["pb_same_pref"] = _df["pb_same_pref"] 
    df["pb_same_v_pref"] = _df["pb_same_v_pref"]
    df["pb_same_genreprice"] = _df["pb_same_genreprice"] 
    df["pb_same_v_genreprice"] = _df["pb_same_v_genreprice"]
    
    df["past_key"] = _df["past_key"]
    df["zprice"] = _df["zprice"]

    df = pd.concat([df, _df[Utility.bin_names_buy]], axis=1)
    df = pd.concat([df, _df[Utility.bin_names_visit]], axis=1)

    return df

def to_xgbdata(_df):
    """Convert xgb values dataframe into dictionary of labels, weights and data."""
    labels = _df.values[:,0]
    weights = _df.values[:,1]
    data = _df.values[:,2:]
    return {"labels":labels, "weights":weights, "data": data} 

create_input_and_write(is_train=False)
create_input_and_write(is_train=True)

LOG.info("finished")