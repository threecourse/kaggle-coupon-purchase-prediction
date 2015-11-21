"""
    This file create (user, coupon) pairs in test period which appeared in visit_log.
"""

import pandas as pd
import numpy as np
from util_logger import get_logger
LOG = get_logger()
LOG.info("start b20")

visit_te = pd.read_pickle("../model/visit_te.pkl")

# remove duplication
visit_test_observed = (visit_te.groupby(["USER_ID", "COUPON_ID"]).size().reset_index().drop(0, axis=1) )
visit_test_observed.to_pickle("../model/visit_test_observed.pkl")

LOG.info("finished")
