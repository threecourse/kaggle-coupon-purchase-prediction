"""
   This file calculates probability of purchase for each user about genres and spotpref.
   - Numerator is count of purchased coupons.
   - Denominator is count of coupons which are in active periods.
     - Active periods are periods where any purchase by the user occured.
"""

import pandas as pd
import numpy as np
from util import Utility, Grouping, Process
from util_logger import get_logger
LOG = get_logger()
LOG.info("start b11a")

# load files
detail_tr = pd.read_pickle("../model/detail_tr.pkl")
coupon_tr = pd.read_pickle("../model/coupon_tr.pkl")
users = pd.read_pickle("../model/users.pkl")

# calculate numer and denom --------------------------------

# remove duplicate
bought = detail_tr.copy()
bought = Grouping.to_group(bought, ["USER_ID", "COUPON_ID"], False)

# denom
active = bought.copy()
active = active.merge(coupon_tr, on = ["COUPON_ID"])
active = Grouping.to_group(active, ["USER_ID","period"], False)

cpntr2 = Grouping.to_group(coupon_tr,["pref","genre","period"], True)

denom = active.copy()
denom = denom.merge(users, on="USER_ID")
denom = denom[["USER_ID","user_pref", "period"]]
denom = denom.merge(cpntr2, on = ["period"])

# numer
bought2 = bought.copy()
bought2 = bought2.merge(users, on = ["USER_ID"])
bought2 = bought2.merge(coupon_tr, on = ["COUPON_ID"])
bought2 = Grouping.to_group(bought2, ["USER_ID", "user_pref", "pref", "genre", "period"], True)

numer = bought2[["USER_ID","pref","genre","count", "period"]].copy()

# add information of spotpref
denom["spotpref"] = np.where(denom.genre.isin(Utility.spot_genre), denom.pref, "NN")
numer["spotpref"] = np.where(numer.genre.isin(Utility.spot_genre), numer.pref, "NN")

# probablity dataframe ------------------------------------

# create pivoted probability dataframe
pivot_genre = Grouping.to_pivotdf(numer, denom, "genre")
pivot_pref = Grouping.to_pivotdf(numer, denom, "spotpref")
pivot_genre_period = Grouping.to_pivotdf_period(numer, denom, "genre")
pivot_pref_period =  Grouping.to_pivotdf_period(numer, denom, "spotpref")

# write
pivot_genre.to_pickle("../model/pivot_genre.pkl")
pivot_pref.to_pickle("../model/pivot_pref.pkl")
pivot_genre_period.to_pickle("../model/pivot_genre_period.pkl")
pivot_pref_period.to_pickle("../model/pivot_pref_period.pkl")

LOG.info("finished")

