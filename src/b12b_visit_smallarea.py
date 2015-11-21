"""
   This file calculates probability of visiting for each user about small_area.
   - Numerator is count of visited coupons.
   - Denominator is count of coupons which are in active periods.
     - Active periods are periods where any visit by the user is observed.
"""

import pandas as pd
import numpy as np
from util import Utility, Grouping
from util_logger import get_logger
LOG = get_logger()
LOG.info("start b12b")

# load files
visit_tr = pd.read_pickle("../model/visit_tr.pkl")
coupon_tr = pd.read_pickle("../model/coupon_tr.pkl")

# calculate numer and denom --------------------------------

# remove duplicate
visit_tr2 = (visit_tr.groupby(["USER_ID", "COUPON_ID"]).size().reset_index().drop(0, axis=1) )

# add information
visit_tr2["small_area"] = Grouping.lookup_coupon_element(visit_tr2, coupon_tr, "small_area")
visit_tr2["genre"] = Grouping.lookup_coupon_element(visit_tr2, coupon_tr, "genre")
visit_tr2["period"] = Grouping.lookup_coupon_element(visit_tr2, coupon_tr, "period")

# group to reduce calculation load
visit = Grouping.to_group(visit_tr2, ["USER_ID", "genre", "small_area", "period"], True)
candidate = Grouping.to_group(coupon_tr, ["genre", "small_area", "period"], True)
active = Grouping.to_group(visit_tr2, ["USER_ID", "period"], False)

# numer
numer = visit.copy()
numer = Grouping.to_group_count(numer, ["USER_ID", "genre", "small_area", "period"])

# denom
denom = candidate.merge(active, on="period")
denom = Grouping.to_group_count(denom, ["USER_ID", "genre", "small_area", "period"])

# exclude non-spot genre and same name with prefecture
numer_mask  = (numer.genre.isin(Utility.spot_genre) & ~(numer.small_area.isin(Utility.prefs)) )
numer = numer[numer_mask]
numer = numer.rename(columns={"small_area":"sarea"})

denom_mask  = (denom.genre.isin(Utility.spot_genre) & ~(denom.small_area.isin(Utility.prefs)) )
denom = denom[denom_mask]
denom = denom.rename(columns={"small_area":"sarea"})

# probablity dataframe ------------------------------------

# create pivoted probability dataframe
visit_pivot_sarea = Grouping.to_pivotdf(numer, denom, "sarea")
visit_pivot_sarea_period = Grouping.to_pivotdf_period(numer, denom, "sarea")

# change column names
visit_pivot_sarea.rename(columns=lambda c: "v_{}".format(c), inplace=True)
visit_pivot_sarea_period.rename(columns=lambda c: "v_{}".format(c), inplace=True)

# write
visit_pivot_sarea.to_pickle("../model/visit_pivot_sarea.pkl")
visit_pivot_sarea_period.to_pickle("../model/visit_pivot_sarea_period.pkl")

LOG.info("finished")
