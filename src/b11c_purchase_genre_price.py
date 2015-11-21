"""
   This file calculates probability of purchase for each user about genre-price.
   - Numerator is count of purchased coupons.
   - Denominator is count of coupons which are in active periods.
     - Active periods are periods where any purchase by the user occured.
"""

import pandas as pd
import numpy as np
from util import Utility, Grouping
from util_logger import get_logger
LOG = get_logger()
LOG.info("start b11c")

# load files
detail_tr = pd.read_pickle("../model/detail_tr.pkl")
coupon_tr = pd.read_pickle("../model/coupon_tr.pkl")
users = pd.read_pickle("../model/users.pkl")

# add genre-price information
coupons_price = pd.read_pickle("../model/coupons_price.pkl")
coupon_tr = coupon_tr.merge(coupons_price, on="COUPON_ID")

# calculate numer and denom --------------------------------

# remove duplicate
bought = detail_tr.copy()
bought = Grouping.to_group(bought, ["USER_ID", "COUPON_ID"], False)

# denom
active = bought.copy()
active = active.merge(coupon_tr, on = ["COUPON_ID"])
active = Grouping.to_group(active, ["USER_ID","period"], False)

cpntr2 = Grouping.to_group(coupon_tr,["genreprice","period"], True)

denom = active.copy()
denom = denom.merge(users, on="USER_ID")
denom = denom[["USER_ID", "period"]]
denom = denom.merge(cpntr2, on = ["period"])
denom = denom[["USER_ID","genreprice","count", "period"]].copy()

# numer
bought2 = bought.copy()
bought2 = bought2.merge(users, on = ["USER_ID"])
bought2 = bought2.merge(coupon_tr, on = ["COUPON_ID"])
bought2 = Grouping.to_group(bought2, ["USER_ID", "genreprice", "period"], True)

numer = bought2[["USER_ID","genreprice", "count", "period"]].copy()

# probablity dataframe ------------------------------------

# create pivoted probability dataframe
pivot_genreprice = Grouping.to_pivotdf(numer, denom, "genreprice")
pivot_genreprice_period =  Grouping.to_pivotdf_period(numer, denom, "genreprice")

# write
pivot_genreprice.to_pickle("../model/pivot_genreprice.pkl")
pivot_genreprice_period.to_pickle("../model/pivot_genreprice_period.pkl")

LOG.info("finished")

