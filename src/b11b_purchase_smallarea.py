"""
   This file calculates probability of purchase for each user about small_area.
   - Numerator is count of purchased coupons.
   - Denominator is count of coupons which are in active periods.
     - Active periods are periods where any purchase by the user occured.
"""

import pandas as pd
import numpy as np
from util import Utility, Grouping
from util_logger import get_logger
LOG = get_logger()
LOG.info("start b11b")

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

cpntr2 = Grouping.to_group(coupon_tr,["small_area","genre","period"], True)

denom = active.copy()
denom = denom.merge(users, on="USER_ID")
denom = denom[["USER_ID", "period"]]
denom = denom.merge(cpntr2, on = ["period"])
denom = denom[["USER_ID","small_area","genre","count", "period"]].copy()

# numer
bought2 = bought.copy()
bought2 = bought2.merge(users, on = ["USER_ID"])
bought2 = bought2.merge(coupon_tr, on = ["COUPON_ID"])
bought2 = Grouping.to_group(bought2, ["USER_ID", "small_area", "genre", "period"], True)

numer = bought2[["USER_ID","small_area","genre","count", "period"]].copy()

# exclude non-spot genre and same name with prefecture
denom_mask = ( denom.genre.isin(Utility.spot_genre) & ~(denom.small_area.isin(Utility.prefs)) )
denom = denom[denom_mask]
denom = denom.rename(columns={"small_area":"sarea"})

numer_mask = ( numer.genre.isin(Utility.spot_genre) & ~(numer.small_area.isin(Utility.prefs)) )
numer = numer[numer_mask]
numer = numer.rename(columns={"small_area":"sarea"})

# probablity dataframe ------------------------------------

# create pivoted probability dataframe
pivot_sarea = Grouping.to_pivotdf(numer, denom, "sarea")
pivot_sarea_period =  Grouping.to_pivotdf_period(numer, denom, "sarea")

# write
pivot_sarea.to_pickle("../model/pivot_sarea.pkl")
pivot_sarea_period.to_pickle("../model/pivot_sarea_period.pkl")

LOG.info("finished")

