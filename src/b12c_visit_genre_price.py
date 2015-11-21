"""
   This file calculates probability of visiting for each user about genre-price.
   - Numerator is count of visited coupons.
   - Denominator is count of coupons which are in active periods.
     - Active periods are periods where any visit by the user is observed.
"""

import pandas as pd
import numpy as np
from util import Utility, Grouping
from util_logger import get_logger
LOG = get_logger()
LOG.info("start b12c")

# load files
visit_tr = pd.read_pickle("../model/visit_tr.pkl")
coupon_tr = pd.read_pickle("../model/coupon_tr.pkl")

# add genre-price information
coupons_price = pd.read_pickle("../model/coupons_price.pkl")
coupon_tr = coupon_tr.merge(coupons_price, on="COUPON_ID")

# calculate numer and denom --------------------------------

# remove duplicate
visit_tr2 = visit_tr.groupby(["USER_ID", "COUPON_ID"]).size().reset_index().drop(0, axis=1)

# add information
visit_tr2["genreprice"] = Grouping.lookup_coupon_element(visit_tr2, coupon_tr,"genreprice")
visit_tr2["period"] = Grouping.lookup_coupon_element(visit_tr2, coupon_tr, "period")

# group to reduce calculation load
visit = Grouping.to_group(visit_tr2, ["USER_ID", "genreprice", "period"], True)
candidate = Grouping.to_group(coupon_tr, ["genreprice", "period"], True)
active = Grouping.to_group(visit_tr2, ["USER_ID", "period"], False)

# numer
numer = visit.copy()
numer = Grouping.to_group_count(numer, ["USER_ID", "genreprice", "period"])

# denom
denom = candidate.merge(active, on="period")
denom = Grouping.to_group_count(denom, ["USER_ID", "genreprice", "period"])

# probablity dataframe ------------------------------------

# create pivoted probability dataframe
visit_pivot_genreprice = Grouping.to_pivotdf(numer, denom, "genreprice")
visit_pivot_genreprice_period = Grouping.to_pivotdf_period(numer, denom, "genreprice")

# change column names
visit_pivot_genreprice.rename(columns=lambda c: "v_{}".format(c), inplace=True)
visit_pivot_genreprice_period.rename(columns=lambda c: "v_{}".format(c), inplace=True)

# write
visit_pivot_genreprice.to_pickle("../model/visit_pivot_genreprice.pkl")
visit_pivot_genreprice_period.to_pickle("../model/visit_pivot_genreprice_period.pkl")

LOG.info("finished")
