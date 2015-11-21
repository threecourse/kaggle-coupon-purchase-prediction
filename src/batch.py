"""
    This file is a batch script to run all.
"""

from subprocess import check_call
import os
import re

def run_all(runtype, version):

    # delete old files before starting
    delete_files()

    # preparation
    run_ipython("a00_prepare.py", [runtype, version])

    # feature engineering
    run_ipython("b00_price.py", [runtype, version])
    run_ipython("b10_location.py", [runtype, version])
    run_ipython("b11a_purchase.py", [runtype, version])
    run_ipython("b11b_purchase_smallarea.py", [runtype, version])
    run_ipython("b11c_purchase_genre_price.py", [runtype, version])
    run_ipython("b12a_visit.py", [runtype, version])
    run_ipython("b12b_visit_smallarea.py", [runtype, version])
    run_ipython("b12c_visit_genre_price.py", [runtype, version])
    run_ipython("b13_couponkey_popularity.py", [runtype, version])
    run_ipython("b14_past_purchase_key.py", [runtype, version])
    run_ipython("b15_area.py", [runtype, version])
    run_ipython("b20_visit_log.py", [runtype, version])

    # train and predict
    run_ipython("c00_selection.py", [runtype, version])
    run_ipython("d00_create_vwtxt.py", [runtype, version]) 
    run_ipython("d01_create_xgbdata.py", [runtype, version])
    run_bash("e00_vw.sh", [runtype, version])
    run_ipython("e01_xgb.py", [runtype, version])
    
    # create submission and manage files
    run_ipython("f00_create_submission.py", [runtype, version])
    run_ipython("g00_filemanage.py", [runtype, version])

def run_ipython(fname, args):
    """Run python script with args."""
    check_call(["ipython", fname] + args)

def run_bash(fname, args):
    """Run bash script with args."""
    check_call(["bash", fname] + args)

def delete_files():
    """Delete files in model directory and .pyc files."""
    # delete files in model
    for root, dirs, files in os.walk("../model"):
        for name in files:
            os.remove(os.path.join(root, name))

    # delete pyc
    files = [f for f in os.listdir('.') if os.path.isfile(f)]
    for name in files :
        if re.match(".*\.pyc", name) : os.remove(name)
    
run_all("production", "p56Fin6")
