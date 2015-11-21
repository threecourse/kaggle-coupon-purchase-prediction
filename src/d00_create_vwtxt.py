"""
    This file creates input file for vowpal wabbit.
"""

import pandas as pd
import numpy as np
import gc
import multiprocessing as mp
from sklearn.externals import joblib
from util import Utility, Process
from util_logger import get_logger
from dxx_dataframe_preprocess import create_dataframes_with_features
LOG = get_logger()
LOG.info("start d00")

def create_input_and_write(is_train):

    if is_train:
        train_ids = pd.read_pickle("../model/train_ids.pkl")
        lines = create_input_multiprocess(train_ids)
        lines.to_csv("../model/train.vwtxt", index=False)
        
        LOG.info("finished train write")
    else:
        test_ids = pd.read_pickle("../model/test_ids.pkl")
        lines = create_input_multiprocess(test_ids)
        lines.to_csv("../model/test.vwtxt", index=False)
        
        LOG.info("finished test write")

def create_input_multiprocess(ids):
    """Create input in multi process."""

    threads = 8 
    p = mp.Pool(threads)
    pool_results = p.map(create_input, np.array_split(ids, threads * 2))
    p.close()
    p.join()

    ret = pd.concat(pool_results)

    pool_results = None
    gc.collect()

    return ret

def create_input(ids):
    """Create input, which is series of vowpal wabbit string."""

    df = ids.copy()
    df = create_dataframes_with_features(df)
    lines = to_vwlines(df)

    # release memory
    df = None   
    gc.collect()
    return lines

def to_vwlines(_df):
    """Convert dataframe into series of vowpal wabbit string."""

    LOG.info("start to_vwlines")

    lines_YABCD = to_str_series(_df, "y") + " " + to_str_series(_df, "count")
    
    lines_YABCD += " |A " + _df.genre
    lines_YABCD += " |B " + _df.user_pref
    lines_YABCD += " |C" 
    lines_YABCD += np.where(_df.pref0, " prefme_T", " prefme_F") 
    lines_YABCD += np.where(_df.pref24, " pref24_T", " pref24_F") 
    lines_YABCD += " |D " + _df.pref

    lines_I = to_emptystr_series(_df) 
    lines_I += " |I"
    lines_I += to_floatstr_series(_df, "lnpop")

    lines_N = to_emptystr_series(_df) 
    lines_N += " |N"
    lines_N += to_floatstr_series(_df, "d2")
    lines_N += " d2bin" + to_str_series(_df, "d2bin")
    lines_N += np.where(_df["area"] == 1, " area", "")

    lines_OP = to_emptystr_series(_df) 
    lines_OP += " |O"
    lines_OP += " " + _df.SEX_ID
    lines_OP += " |P"
    lines_OP += to_floatstr_series(_df, "AGE")

    lines_Q = to_emptystr_series(_df) 
    lines_Q += " |Q"
    lines_Q += np.where(_df.pref0, " prefme_T", " prefme_F") 
    lines_Q += np.where(_df.user_prefNN, " prefNN_T", " prefNN_F") 
    lines_Q += " " + to_floatstr_series(_df, "d2")

    lines_RS = to_emptystr_series(_df) 
    lines_RS += " |R"
    lines_RS += to_floatstr_series(_df, "pb_same_sarea")
    lines_RS += to_floatstr_series(_df, "pb_same_v_sarea")
    lines_RS += " |S"
    lines_RS += np.where(_df.past_key > 0, " past_key", "")

    lines_FG = to_emptystr_series(_df) 
    lines_FG += " |F" 
    lines_FG += np.where(_df.spot, " spot_T", " spot_F") 
    lines_FG += " |G" 
    lines_FG += np.where(_df.user_prefNN, " user_prefNN_T", " user_prefNN_F")

    lines_E = ""
    lines_E += " |E"
    for ge in Utility.genres:
        lines_E += to_floatstr_series(_df, ge) 
    
    lines_H =  ""
    lines_H += " |H"
    for pr in Utility.prefs:
        lines_H = lines_H + to_floatstr_series(_df, pr) 

    lines_K = ""
    lines_K += " |K"
    for ge in Utility.v_genres:
        lines_K += to_floatstr_series(_df, ge) 
    lines_K += to_floatstr_series(_df, "zprice")
    lines_K += to_floatstr_series(_df, "pb_same_genreprice")
    lines_K += to_floatstr_series(_df, "pb_same_v_genreprice")

    lines_L = ""
    lines_L += " |L"
    for pr in Utility.v_prefs:
        lines_L += to_floatstr_series(_df, pr) 

    lines = (lines_YABCD + lines_E + lines_FG + lines_H + lines_I + lines_K 
                         + lines_L + lines_N + lines_OP + lines_Q + lines_RS)

    LOG.info("finish to_vwlines")
    return lines

# utility
def to_floatstr_series(_df, col) :
    """Create series of string, representing column name and float value like 'AGE:50.000000'."""

    def to_floatstr(c, x):
        if x == 0: return ""
        return " {0}:{1:.6f}".format(c, x)
    return to_series(_df, col, to_floatstr) 

def to_str_series(_df, col) :
    """Create series of string, representing float value."""
    return to_series(_df, col, lambda c, x: str(x)) 

def to_emptystr_series(_df) :
    """Create series of empty string"""
    return pd.Series("", index=_df.index)

def to_series(_df, col, _func):
    func = np.vectorize(_func)
    return pd.Series(func(col, _df[col]), index=_df.index)

create_input_and_write(is_train=False)
create_input_and_write(is_train=True)

LOG.info("finished")