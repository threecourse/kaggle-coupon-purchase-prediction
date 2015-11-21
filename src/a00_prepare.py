"""
    This file is for preparing data as below.
    - Translate Japanese strings into English.
    - Replace hash strings with ids.
    - Add period to coupons.
"""

import pandas as pd
import numpy as np
from util_logger import get_logger
LOG = get_logger()
LOG.info("a00")

# 0. load files ----------------------------------------

user_list = pd.read_csv("../input/user_list.csv")
coupon_area_train = pd.read_csv("../input/coupon_area_train.csv")
coupon_area_test = pd.read_csv("../input/coupon_area_test.csv")
coupon_list_train  = pd.read_csv("../input/coupon_list_train.csv")
coupon_list_test = pd.read_csv("../input/coupon_list_test.csv")
coupon_detail_train = pd.read_csv("../input/coupon_detail_train.csv")
coupon_visit_train = pd.read_csv("../input/coupon_visit_train.csv")

def csv_to_dict(path):
    df = pd.read_csv(path)
    return dict([(r.jp, r.en) for i, r in df.iterrows()])

dict_SMALLAREA = csv_to_dict("../input/SMALLAREA.csv")
dict_PREF = csv_to_dict("../input/PREF.csv")
dict_GENRE = csv_to_dict("../input/GENRE.csv")
dict_LARGEAREA = csv_to_dict("../input/LARGEAREA.csv")
dict_CAPSULE = csv_to_dict("../input/CAPSULE.csv")

# 1. translate -------------------------------

user_list["pref"] = user_list.PREF_NAME.replace(dict_PREF)
coupon_area_train["pref"] = coupon_area_train.PREF_NAME.replace(dict_PREF)
coupon_area_test["pref"] = coupon_area_test.PREF_NAME.replace(dict_PREF)
coupon_list_train["pref"] = coupon_list_train.ken_name.replace(dict_PREF)
coupon_list_test["pref"] = coupon_list_test.ken_name.replace(dict_PREF)

coupon_list_train["large_area"] = coupon_list_train.large_area_name.replace(dict_LARGEAREA)
coupon_list_test["large_area"] = coupon_list_test.large_area_name.replace(dict_LARGEAREA)

coupon_area_train["small_area"] = coupon_area_train.SMALL_AREA_NAME.replace(dict_SMALLAREA)
coupon_area_test["small_area"] = coupon_area_test.SMALL_AREA_NAME.replace(dict_SMALLAREA)
coupon_list_train["small_area"] = coupon_list_train.small_area_name.replace(dict_SMALLAREA)
coupon_list_test["small_area"] = coupon_list_test.small_area_name.replace(dict_SMALLAREA)

coupon_list_train["capsule"] = coupon_list_train.CAPSULE_TEXT.replace(dict_CAPSULE)
coupon_list_test["capsule"] = coupon_list_test.CAPSULE_TEXT.replace(dict_CAPSULE)

coupon_list_train["genre"] = coupon_list_train.GENRE_NAME.replace(dict_GENRE)
coupon_list_test["genre"] = coupon_list_test.GENRE_NAME.replace(dict_GENRE)

user_list = user_list.drop(["PREF_NAME"], axis=1)
coupon_area_train = coupon_area_train.drop(["PREF_NAME", "SMALL_AREA_NAME"], axis=1)
coupon_area_test = coupon_area_test.drop(["PREF_NAME", "SMALL_AREA_NAME"], axis=1)
coupon_list_train = coupon_list_train.drop(["ken_name", "large_area_name","small_area_name","CAPSULE_TEXT","GENRE_NAME"], axis=1)
coupon_list_test = coupon_list_test.drop(["ken_name", "large_area_name","small_area_name","CAPSULE_TEXT","GENRE_NAME"], axis=1)
coupon_detail_train = coupon_detail_train.drop(["SMALL_AREA_NAME"], axis=1)

# 2. replace hash string ---------------------------------
user_ids = user_list.USER_ID_hash
coupon_train_ids = coupon_list_train.COUPON_ID_hash

# for test coupon, added 1000000 to IDs
coupon_test_ids = coupon_list_test.COUPON_ID_hash
coupon_test_ids.rename(lambda x : x + 1000000, inplace=True)
coupon_ids = pd.concat([coupon_train_ids, coupon_test_ids])

def replace_hash(hashes, hash_id_table):
    replace_table = pd.Series(hash_id_table.index, index=hash_id_table.values)
    return replace_table[hashes].values

# replace hash and drop extra columns
detail = coupon_detail_train.copy()
detail["USER_ID"] = replace_hash(detail.USER_ID_hash, user_ids)
detail["COUPON_ID"] = replace_hash(detail.COUPON_ID_hash, coupon_ids)
detail = detail[["USER_ID", "COUPON_ID"]]

coupon_train = coupon_list_train.copy()
coupon_train["COUPON_ID"] = replace_hash(coupon_train.COUPON_ID_hash, coupon_ids)
coupon_train.drop("COUPON_ID_hash", axis=1, inplace=True)

coupon_test = coupon_list_test.copy()
coupon_test["COUPON_ID"] = replace_hash(coupon_test.COUPON_ID_hash, coupon_ids)
coupon_test.drop("COUPON_ID_hash", axis=1, inplace=True)

users = user_list.copy()
users["USER_ID"] = replace_hash(users.USER_ID_hash, user_ids)
users.drop("USER_ID_hash", axis=1, inplace=True)

visit_train = coupon_visit_train[coupon_visit_train.VIEW_COUPON_ID_hash.isin(coupon_train_ids)].copy()
visit_train["USER_ID"] = replace_hash(visit_train.USER_ID_hash, user_ids)
visit_train["COUPON_ID"] = replace_hash(visit_train.VIEW_COUPON_ID_hash, coupon_train_ids)
visit_train = visit_train[["USER_ID","COUPON_ID","PURCHASE_FLG"]]

visit_test = coupon_visit_train[coupon_visit_train.VIEW_COUPON_ID_hash.isin(coupon_test_ids)].copy()
visit_test["USER_ID"] = replace_hash(visit_test.USER_ID_hash, user_ids)
visit_test["COUPON_ID"] = replace_hash(visit_test.VIEW_COUPON_ID_hash, coupon_test_ids)
visit_test = visit_test[["USER_ID","COUPON_ID","PURCHASE_FLG"]]

coupon_area = pd.concat([coupon_area_train, coupon_area_test], ignore_index=True)
coupon_area["COUPON_ID"] = replace_hash(coupon_area.COUPON_ID_hash, coupon_ids)
coupon_area.drop("COUPON_ID_hash", axis=1, inplace=True)

# 3-1. add period ------------------------------

def to_period(_DISPFROM):
    """Add period. Period is weeks to the date test period starts"""
    test_start = pd.to_datetime("2012-06-24")
    days_to_start = (test_start - pd.to_datetime(_DISPFROM)).dt.days 
    period = np.floor_divide(days_to_start, 7)
    return period

coupon_train["period"] = to_period(coupon_train.DISPFROM)

# test period is -1
coupon_test["period"] = -1

# concat train and test
coupons = pd.concat([coupon_train, coupon_test])

# 3-2. rename pref -----------------------------
users = users.rename(columns={"pref":"user_pref"})
users["user_pref"] = users["user_pref"].fillna("NN")

# 4. write ---------------------------------------

# ids -----
user_ids.to_pickle("../model/user_ids.pkl")
coupon_ids.to_pickle("../model/coupon_ids.pkl")

# information -----
coupons.to_pickle("../model/coupons.pkl")
users.to_pickle("../model/users.pkl")
coupon_area.to_pickle("../model/coupon_area.pkl")

# information of train coupons -----
coupon_tr = coupon_train
detail_tr = detail
visit_tr = visit_train

detail_tr.to_pickle("../model/detail_tr.pkl")
coupon_tr.to_pickle("../model/coupon_tr.pkl")
visit_tr.to_pickle("../model/visit_tr.pkl")

# information of test coupons -----
visit_te = visit_test
coupon_te = coupon_test

visit_te.to_pickle("../model/visit_te.pkl")
coupon_te.to_pickle("../model/coupon_te.pkl")

LOG.info("finished")