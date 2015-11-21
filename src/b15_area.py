"""
    This file creates (COUPON_ID, pref) pairs whose pref appeared in coupon_area_train/coupon_area_test.
"""

import pandas as pd
import numpy as np
from util_logger import get_logger
LOG = get_logger()
LOG.info("start b15")

# load files
coupon_area = pd.read_pickle("../model/coupon_area.pkl")

# remove duplication
coupon_areas_feature = coupon_area.groupby(["COUPON_ID", "pref"]).size().reset_index().drop(0, axis=1)
coupon_areas_feature["area"] = 1
coupon_areas_feature = coupon_areas_feature.rename(columns={"pref":"user_pref"})
coupon_areas_feature.to_pickle("../model/coupon_areas_feature.pkl")

LOG.info("finished")
