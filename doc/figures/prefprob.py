"""
    This file is for creating table in model documentation.    
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

user_ids = pd.read_pickle("../model/user_ids.pkl")
coupon_ids = pd.read_pickle("../model/coupon_ids.pkl")

detail = pd.read_pickle("../model/detail.pkl")
coupons = pd.read_pickle("../model/coupons.pkl")
users = pd.read_pickle("../model/users.pkl")

detail = detail.groupby(["COUPON_ID","USER_ID"]).size().reset_index().drop(0, axis=1)
detail = detail.merge(coupons, on="COUPON_ID")
detail = detail.merge(users, on="USER_ID")
detail = detail[detail.user_pref != "NN"]
detail["same_pref"] = (detail.pref == detail.user_pref)

active = detail.groupby(["USER_ID","period"]).size().reset_index().drop(0, axis=1)

numer = detail.groupby(["genre","same_pref"]).size().reset_index().rename(columns={0:"count"})

denom = coupons.groupby(["pref","genre","period"]).size().reset_index().rename(columns={0:"count"})
denom = denom.merge(active, on="period")
denom = denom.merge(users[["USER_ID","user_pref"]], on="USER_ID")
denom = denom[denom.user_pref != "NN"]
denom["same_pref"] = denom.pref == denom.user_pref
denom = denom.groupby(["genre","same_pref"])["count"].sum().reset_index()

all = denom.merge(numer, on=["genre","same_pref"], suffixes=("_d","_n"), how="left")
all["prob"] = all["count_n"] / all["count_d"]

pivot = all.pivot_table(index="genre", values="prob", columns="same_pref")
pivot["ratio"] = np.floor(pivot[True] / pivot[False] * 100) / 100

print pivot