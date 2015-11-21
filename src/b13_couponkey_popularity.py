"""
    This file inferred popularity of each coupon by couponkey and genre.
"""

import pandas as pd
import numpy as np
from util import Process
from util_logger import get_logger
LOG = get_logger()
LOG.info("start b13")

# load files
detail_tr = pd.read_pickle("../model/detail_tr.pkl")
coupons = pd.read_pickle("../model/coupons.pkl")
coupon_tr = pd.read_pickle("../model/coupon_tr.pkl")
users = pd.read_pickle("../model/users.pkl")

# count purchases for each coupon ---------------------------

# exclude duplication
detail_tr = detail_tr.groupby(["COUPON_ID","USER_ID"]).size().reset_index().drop(0, axis=1)

detail_cp = detail_tr.groupby("COUPON_ID").size().reset_index().rename(columns={0:"count"})
detail_cp = detail_cp.merge(coupon_tr, on="COUPON_ID")
detail_cp = detail_cp.sort("count")
detail_cp["key1"] = Process.to_key1(detail_cp)

# calculate couponkey and genre popularility ----------------

# couponkey popularity, calculated as mean 
popular_key = detail_cp.groupby("key1")["count"].agg([np.mean, np.size])
popular_key = popular_key.rename(columns={"mean":"key_mean", "size":"key_size"})

# genre popularity, calculated as mean 
popular_genre = detail_cp.groupby("genre")["count"].agg([np.mean, np.size])
popular_genre = popular_genre.rename(columns={"mean":"key_mean", "size":"key_size"})

# couponkey popularity, exclude couponkey of only one sample
popular_key_train = popular_key[popular_key["key_size"] > 1]

# set popularility for each coupon ----------------------
coupons["key1"] = Process.to_key1(coupons)

# popularitly is set by couponkey for train coupon, if couponkey has two or more samples
# popularitly is set by couponkey for test coupon
pop_key1_train = popular_key_train.loc[coupons["key1"] ]["key_mean"].values
pop_key1_test  = popular_key.loc[coupons["key1"] ]["key_mean"].values
is_test = coupons.period < 0
pop_key1 = np.where(is_test, pop_key1_test, pop_key1_train)

# set popularitly
# if nan, genre popularility is set
pop_genre = popular_genre.loc[coupons["genre"] ]["key_mean"].values
coupons["pop"] = np.where(np.isnan(pop_key1), pop_genre, pop_key1)

# write
coupon_pop = coupons[["COUPON_ID","pop"]]
coupon_pop.to_pickle("../model/coupon_pop.pkl")

LOG.info("finished")