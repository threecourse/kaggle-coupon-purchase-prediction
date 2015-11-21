"""
    This file creates features about locations.
"""

import pandas as pd
import numpy as np
from util import Utility
from util_logger import get_logger
import math
from scipy import stats

LOG = get_logger()
LOG.info("start b10")

def haversine(lo1, la1, lo2, la2):
    """Calculate haversine distance"""
    lo1, la1, lo2, la2 = map(math.radians, [lo1, la1, lo2, la2])
    a1 = math.sin( (la2 - la1) / 2) ** 2
    a2 = math.cos(la1) * math.cos(la2) * math.sin( (lo2 - lo1) / 2) ** 2
    return 2 * math.asin(math.sqrt(a1 + a2)) * 6371

def csv_to_dict(path):
    df = pd.read_csv(path)
    return dict([(r.jp, r.en) for i, r in df.iterrows()])

def create_location3():

    # add near_prefs - list ordered by distance from pref
    loc1 = locations.copy()
    loc1["near_prefs"] = None

    for i, r in loc1.iterrows():
        loc2 = locations.copy()
        loc2["LA"] = r.LATITUDE
        loc2["LO"] = r.LONGITUDE
        loc2["d"] = map(haversine, loc2.LONGITUDE, loc2.LATITUDE, loc2.LO, loc2.LA)
        near = loc2.sort("d").pref.values
        near = list(near)
        loc1.loc[i, "near_prefs"] = near

    # create dataframe of user_pref, pref, dist. 
    # dist is the order of pref by the distance from user_pref.
    records = []
    for i, r in loc1.iterrows():
        upref = r.pref
        near_prefs = r.near_prefs
        for dist, pref in enumerate(near_prefs):
            records += [(upref, pref, dist)]

    for pref in locations.pref:
        records += [("NN", pref, -1)]

    df = pd.DataFrame(records, columns=["user_pref","pref","dist"])

    # add LONGITUDE and LATITUDE
    df = df.merge(locations[["pref","LONGITUDE","LATITUDE"]], left_on="pref", right_on="pref", how="left")
    df = df.merge(locations[["pref","LONGITUDE","LATITUDE"]], left_on="user_pref", right_on="pref", how="left", suffixes=("", "_u"))

    # add haversine distance
    df["d"] = map(haversine, df["LONGITUDE"], df["LATITUDE"], df["LONGITUDE_u"], df["LATITUDE_u"])
   
    # cut off by 250 kilometer
    cut_off = 250
    df["d2"] = df["d"].copy()
    df["d2"].fillna(cut_off, inplace=True)
    df["d2"] = np.minimum(cut_off, df["d2"])

    # binned by 50 kilometer
    df["d2bin"] = np.floor_divide(df["d2"], 50) + 1
    df["d2bin"] = np.where(df["d2"] == 0, 0, df["d2bin"])
    df["d2bin"] = np.where(df["user_pref"] == "NN", 99, df["d2bin"])

    return df[["user_pref","pref","dist","d", "d2", "d2bin"]]

def create_pref_to_id():
    """Order prefectures along intuitive axis, for label encoding to work well."""

    loc4 = locations.copy()
    
    # z is position along the intuitive axis
    x = locations.LONGITUDE
    y = locations.LATITUDE
    slope, intercept, r_value, _, _ = stats.linregress(x, y)
    z = - (x + slope * y)

    loc4["z"] = z

    # add large_area
    large_areas = coupons.groupby(["pref","large_area"]).size().reset_index()[["pref","large_area"]]
    loc4 = loc4.merge(large_areas, on="pref")
    loc4["z_large_area"] = loc4.groupby("large_area")["z"].transform(np.mean)

    # set id by large_area and z
    loc4 = loc4.sort(["z_large_area", "z"]).reset_index(drop=True)
    loc4["prefid"] = loc4.index.values

    pref_to_id = pd.Series(loc4.prefid.values, index=loc4.pref)

    return pref_to_id

# load files
locations = pd.read_csv("../input/prefecture_locations.csv")
dict_PREF = csv_to_dict("../input/PREF.csv")
locations["pref"] = locations.PREF_NAME.replace(dict_PREF)
locations.drop(["PREF_NAME", "PREFECTUAL_OFFICE"], axis=1, inplace=True)
coupons = pd.read_pickle("../model/coupons.pkl")

# create features
locations3  = create_location3()
pivot_pref_distbin = pd.pivot_table(locations3, "d2bin", index="pref", columns="user_pref", aggfunc="sum")
pivot_pref_distbin = pivot_pref_distbin[Utility.prefs]
pref_to_id = create_pref_to_id()

# write
locations3.to_pickle("../model/locations3.pkl")
pivot_pref_distbin.to_pickle("../model/pivot_pref_distbin.pkl")
pref_to_id.to_pickle("../model/pref_to_id.pkl")

LOG.info("finished")
