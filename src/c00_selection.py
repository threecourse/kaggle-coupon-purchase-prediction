"""
    This file generates, selects and samples train records and generates test records.
    Here, record consists of USER_ID, COUPON_ID, weight and label.
"""

import pandas as pd
import numpy as np
from util import Utility, Grouping
from util_logger import get_logger
LOG = get_logger()
LOG.info("start c00")

# load files
users = pd.read_pickle("../model/users.pkl")
coupons = pd.read_pickle("../model/coupons.pkl")
coupon_tr = pd.read_pickle("../model/coupon_tr.pkl")
coupon_te = pd.read_pickle("../model/coupon_te.pkl")
detail_tr = pd.read_pickle("../model/detail_tr.pkl")

# train data ---------------------------------

# exclude non_spot genre from training data
coupon_tr2 = coupon_tr[coupon_tr.genre.isin(Utility.spot_genre)].copy()
detail_tr2 = detail_tr[detail_tr.COUPON_ID.isin(coupon_tr2.COUPON_ID)].copy()

# user/coupon pairs, purchase occured
bought = detail_tr2.copy()
bought = Grouping.to_group(bought, ["USER_ID", "COUPON_ID"], False)

# user/period pairs, purchase occured in the period
active = bought.copy()
active = active.merge(coupon_tr2[["COUPON_ID","period"]], on = ["COUPON_ID"])
active = Grouping.to_group(active, ["USER_ID", "period"], False)

def random_index(all_count, sample_count, seed):
    """Create integer list randomly, sample_count out of all_count."""
    np.random.seed(seed)
    rd = np.random.rand(all_count)
    idxes = np.argsort(rd)
    return idxes[:sample_count]

# negative samples - coupons * users active in the period
traind = coupon_tr2[["COUPON_ID","period"]].copy()
traind = traind.merge(active, on = ["period"])
traind["y"] = -1
traind["count"] = 1
LOG.info("traind {}, sum {}".format(len(traind), traind["count"].sum())) 

# apply sampling because rows are too many
samples = np.minimum(len(traind), 5000000)
weight = len(traind) / float(samples)
LOG.info("weight multiplier {}".format(weight)) 
traind = traind.iloc[random_index(len(traind), samples, seed=1234)]
traind = traind.reset_index(drop=True)
traind["count"] *= weight

# remove extra columns from negative samples
traind = traind[["USER_ID","COUPON_ID","y","count"]]
LOG.info( "traind {}, sum {}".format(len(traind), traind["count"].sum()) )

# positive samples - user/coupon pairs, purchase occured
trainn = bought.copy()
trainn["y"] = 1
trainn["count"] = 1
LOG.info("trainn {}, sum {}".format(len(trainn), trainn["count"].sum())) 

# adjust balance between positive and negative samples
multiplier_p = float(traind["count"].sum()) / trainn["count"].sum()
LOG.info("weight multiplier positive {}".format(multiplier_p))
trainn["count"] *= multiplier_p 

# remove extra columns from positive samples
trainn = trainn[["USER_ID","COUPON_ID","y","count"]]
LOG.info( "trainn {}, sum {}".format(len(trainn), trainn["count"].sum()) )

# concat negative and positive samples
train_ids = pd.concat([trainn, traind], ignore_index=True)

# randomize order
train_ids = train_ids.iloc[random_index(len(train_ids), len(train_ids), seed=12345)]
train_ids = train_ids.reset_index(drop=True)
LOG.info( "train {}, sum {} - all".format(len(train_ids), train_ids["count"].sum()) )

train_ids.to_pickle("../model/train_ids.pkl")

# test data ---------------------------------

# all users are assumed active in test period
active_te = users[["USER_ID"]].copy()
active_te["period"] = -1

test = coupon_te[["COUPON_ID","period"]].copy()
test = test.merge(active_te, on = ["period"])
test["y"] =  -1 # dummy
test["count"] = 1 # dummy
test_ids = test[["USER_ID","COUPON_ID","y","count"]]

test_ids.to_pickle("../model/test_ids.pkl")

LOG.info("finished")