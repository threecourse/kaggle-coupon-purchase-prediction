"""
   This file calculates probability of visiting for each user about genres and spotpref.
   - Numerator is count of visited coupons.
   - Denominator is count of coupons which are in active periods.
     - Active periods are periods where any visit by the user is observed.
"""

import pandas as pd
import numpy as np
from util import Utility, Grouping, Process
from util_logger import get_logger
LOG = get_logger()
LOG.info("start b12a")

# load files
visit_tr = pd.read_pickle("../model/visit_tr.pkl")
coupon_tr = pd.read_pickle("../model/coupon_tr.pkl")

# calculate numer and denom --------------------------------

# remove duplicate
visit_tr2 = (visit_tr.groupby(["USER_ID", "COUPON_ID"]).size().reset_index().drop(0, axis=1) )

# add information
visit_tr2["pref"] = Grouping.lookup_coupon_element(visit_tr2, coupon_tr, "pref")
visit_tr2["genre"] = Grouping.lookup_coupon_element(visit_tr2, coupon_tr, "genre")
visit_tr2["period"] = Grouping.lookup_coupon_element(visit_tr2, coupon_tr, "period")

# group to reduce calculation load
visit = Grouping.to_group(visit_tr2, ["USER_ID", "genre", "pref", "period"], True)
candidate = Grouping.to_group(coupon_tr, ["genre", "pref", "period"], True)
active = Grouping.to_group(visit_tr2, ["USER_ID", "period"], False)

# numer
numer = visit.copy()
numer = Grouping.to_group_count(numer, ["USER_ID", "genre", "pref", "period"])

# denom
denom = candidate.merge(active, on="period")
denom = Grouping.to_group_count(denom, ["USER_ID", "genre", "pref", "period"])

# add information of spotpref
numer["spotpref"] = np.where(numer.genre.isin(Utility.spot_genre), numer.pref, "NN")
denom["spotpref"] = np.where(denom.genre.isin(Utility.spot_genre), denom.pref, "NN")

# probablity dataframe ------------------------------------

# create pivoted probability dataframe
visit_pivot_genre = Grouping.to_pivotdf(numer, denom, "genre")
visit_pivot_pref = Grouping.to_pivotdf(numer, denom, "spotpref")
visit_pivot_genre_period = Grouping.to_pivotdf_period(numer, denom, "genre")
visit_pivot_pref_period = Grouping.to_pivotdf_period(numer, denom, "spotpref")

# change column names
visit_pivot_genre.rename(columns=lambda c: "v_{}".format(c), inplace=True)
visit_pivot_pref.rename(columns=lambda c: "v_{}".format(c), inplace=True)
visit_pivot_genre_period.rename(columns=lambda c: "v_{}".format(c), inplace=True)
visit_pivot_pref_period.rename(columns=lambda c: "v_{}".format(c), inplace=True)

# write
visit_pivot_genre.to_pickle("../model/visit_pivot_genre.pkl")
visit_pivot_pref.to_pickle("../model/visit_pivot_pref.pkl")
visit_pivot_genre_period.to_pickle("../model/visit_pivot_genre_period.pkl")
visit_pivot_pref_period.to_pickle("../model/visit_pivot_pref_period.pkl")

LOG.info("finished")
