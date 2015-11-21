"""
    This file counts purchases by each (user, couponkey) pairs. 
"""

import pandas as pd
import numpy as np
from util import Grouping, Process
from util_logger import get_logger
LOG = get_logger()
LOG.info("start b14")

# load files
users = pd.read_pickle("../model/users.pkl")
coupons = pd.read_pickle("../model/coupons.pkl")
coupon_tr = pd.read_pickle("../model/coupon_tr.pkl")
detail_tr = pd.read_pickle("../model/detail_tr.pkl")

# count purchases per (user, couponkey) 
bought = detail_tr.copy()
bought = Grouping.to_group(bought, ["USER_ID", "COUPON_ID"], False)
bought = bought.merge(coupon_tr, on="COUPON_ID", how='left')
bought["key1"] = Process.to_key1(bought)

# remove duplication
bought = Grouping.to_group(bought, ["USER_ID", "period", "key1"], False) 

# count purchases per (user, couponkey) 
past_buy_key        = Grouping.to_group(bought, ["USER_ID", "key1"], True)
past_buy_key        = past_buy_key.set_index(["USER_ID", "key1"])

past_buy_key_period = Grouping.to_group(bought, ["USER_ID", "key1", "period"], True)
past_buy_key_period = past_buy_key_period.set_index(["USER_ID", "key1", "period"])

# write
past_buy_key.to_pickle("../model/past_buy_key.pkl")
past_buy_key_period.to_pickle("../model/past_buy_key_period.pkl")

LOG.info("finished")
