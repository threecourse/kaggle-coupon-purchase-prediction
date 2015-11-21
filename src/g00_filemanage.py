"""
    This moves a submission file and model files.
"""

from util_logger import get_logger
LOG = get_logger()
LOG.info("started g00")

import os
import sys
import shutil
argvs = sys.argv 
_ , runtype, version = argvs

dirfrom = "../model"
dirto = "../submission"
if not os.path.exists(dirto): os.makedirs(dirto)

# move submission file
shutil.copyfile(os.path.join(dirfrom, "predict_coupons_mix.csv"), 
                os.path.join(dirto,   "submission.csv"  ) )

# move model files
shutil.copyfile(os.path.join(dirfrom, "xgb.model"), 
                os.path.join(dirto,   "xgb.model"))
shutil.copyfile(os.path.join(dirfrom, "train.vwmdl"), 
                os.path.join(dirto,   "train.vwmdl"))