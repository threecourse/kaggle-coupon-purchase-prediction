"""
    This file creates submission file based on predicted values from vowpal wabbit and xgboost.
"""

import pandas as pd
import numpy as np
from util import Utility
import sys
argvs = sys.argv 
_ , runtype, version = argvs

from util_logger import get_logger
LOG = get_logger()
LOG.info("start f00")

# load files ---------------------
test_ids = pd.read_pickle("../model/test_ids.pkl")

users = pd.read_pickle("../model/users.pkl")
coupons = pd.read_pickle("../model/coupons.pkl")

visit_test_observed = pd.read_pickle("../model/visit_test_observed.pkl")
user_ids = pd.read_pickle("../model/user_ids.pkl")
coupon_ids =pd.read_pickle("../model/coupon_ids.pkl")

predict_vw = pd.read_csv("../model/predict.txt", header=None, names=["p"])
predict_xgb = pd.read_csv("../model/xgb_predicted.txt", header=None, names=["prob"])

# calculate predict score ---------------------

# create dataframe of USER_ID, COUPON_ID and predicted values
predict_df = test_ids[["COUPON_ID", "USER_ID"]].copy()
predict_df["p_vw"] = predict_vw["p"].values
predict_df["p_xgb"]  = np.log( predict_xgb["prob"].values / (1 - predict_xgb["prob"].values) )

# predict score is average of vowpal wabbit and xgboost
predict_df["predict_score"] = predict_df["p_vw"]  * 0.5 + predict_df["p_xgb"] * 0.5 

# add information of users and coupons
predict_df = predict_df.merge(users, on="USER_ID")
predict_df = predict_df.merge(coupons, on="COUPON_ID")

# give big score on coupons appeared in visit_log, to choose at first
visit_test_observed["visit_bonus"] = 10000.0
predict_df = predict_df.merge(visit_test_observed, on=["COUPON_ID", "USER_ID"], how="left")
predict_df["visit_bonus"].fillna(0.0, inplace=True)

# give penalty on some genres, to exclude them
predict_df["drop_bonus"] = np.where(predict_df.genre.isin(Utility.spot_genre), 0, -5000.0)

# calculate the final score
predict_df["final_score"] = (predict_df["predict_score"] + predict_df["visit_bonus"] + predict_df["drop_bonus"] )

def create_answer(_predict_df):

    # sort by score and assign rank
    df = _predict_df.copy()
    df = df.sort(["final_score", "DISCOUNT_PRICE", "COUPON_ID"], ascending=[0,1,1])
    df["rank"] = np.arange(0, len(df))

    # sort by USER_ID and rank
    df = df.sort(["USER_ID","rank"])

    # choose coupons for each user
    user_count = len(df.USER_ID.unique())
    coupon_count = len(df.COUPON_ID.unique())
    idxes0 = np.arange(0, user_count) * coupon_count
    p_user_ids = df.iloc[idxes0].USER_ID.values
    p_user_hashes = user_ids.loc[p_user_ids].values

    p_coupon_ids_list = [df.iloc[idxes0 + ii].COUPON_ID.values 
                                             for ii in range(0, 10)]
    p_coupons_hashes_list = [coupon_ids.loc[p_coupon_ids].values 
                                             for p_coupon_ids in p_coupon_ids_list]

    p_coupons_hashes_string = p_coupons_hashes_list[0]
    for p_coupons_hashes in p_coupons_hashes_list[1:]:
       p_coupons_hashes_string = p_coupons_hashes_string + " " + p_coupons_hashes

    # write submission file
    answer_df = pd.DataFrame(zip(p_user_hashes, p_coupons_hashes_string),
                           columns=["USER_ID_hash","PURCHASED_COUPONS"])
    answer_df.to_csv("../model/predict_coupons_mix.csv", index=False)

create_answer(predict_df)

LOG.info("finished")