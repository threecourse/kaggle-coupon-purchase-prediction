"""
    This file has functions to create dataframes with features.    
    These funcations are used in d00_create_vwtxt.py and d01_create_xgbdata.py.
"""

import pandas as pd
import numpy as np
from util import Utility, Process
from util_logger import get_logger
LOG = get_logger()
LOG.info("load dxx")

# load files -------------------------------------
coupons = pd.read_pickle("../model/coupons.pkl")
users = pd.read_pickle("../model/users.pkl")

locations3 = pd.read_pickle("../model/locations3.pkl")

pivot_genre = pd.read_pickle("../model/pivot_genre.pkl")
pivot_pref = pd.read_pickle("../model/pivot_pref.pkl")
pivot_genre_period = pd.read_pickle("../model/pivot_genre_period.pkl")
pivot_pref_period = pd.read_pickle("../model/pivot_pref_period.pkl")
visit_pivot_genre = pd.read_pickle("../model/visit_pivot_genre.pkl")
visit_pivot_pref = pd.read_pickle("../model/visit_pivot_pref.pkl")
visit_pivot_genre_period = pd.read_pickle("../model/visit_pivot_genre_period.pkl")
visit_pivot_pref_period = pd.read_pickle("../model/visit_pivot_pref_period.pkl")

pivot_sarea = pd.read_pickle("../model/pivot_sarea.pkl")
pivot_sarea_period = pd.read_pickle("../model/pivot_sarea_period.pkl")
visit_pivot_sarea = pd.read_pickle("../model/visit_pivot_sarea.pkl")
visit_pivot_sarea_period = pd.read_pickle("../model/visit_pivot_sarea_period.pkl")

past_buy_key = pd.read_pickle("../model/past_buy_key.pkl")
past_buy_key_period = pd.read_pickle("../model/past_buy_key_period.pkl")

pivot_pref_distbin = pd.read_pickle("../model/pivot_pref_distbin.pkl")
coupon_pop = pd.read_pickle("../model/coupon_pop.pkl")
coupon_areas_feature = pd.read_pickle("../model/coupon_areas_feature.pkl")

coupons_price = pd.read_pickle("../model/coupons_price.pkl")

pivot_genreprice = pd.read_pickle("../model/pivot_genreprice.pkl")
pivot_genreprice_period = pd.read_pickle("../model/pivot_genreprice_period.pkl")
visit_pivot_genreprice = pd.read_pickle("../model/visit_pivot_genreprice.pkl")
visit_pivot_genreprice_period = pd.read_pickle("../model/visit_pivot_genreprice_period.pkl")

# features -----------------------

def create_dataframes_with_features(ids):
    _df = ids.copy()
    _df = _df.merge(users, on = ["USER_ID"] , how='left')
    _df = _df.merge(coupons, on = ["COUPON_ID"], how='left')
    _df = _df.merge(locations3, on=["user_pref","pref"], how="left")
    _df = _df.merge(coupons_price, on=["COUPON_ID"], how="left")
   
    _df = add_pivots(_df, pivot_genre, pivot_genre_period)
    _df = add_pivots(_df, pivot_pref, pivot_pref_period)
    _df = add_pivots(_df, visit_pivot_genre, visit_pivot_genre_period)
    _df = add_pivots(_df, visit_pivot_pref, visit_pivot_pref_period)
    _df = add_pivots(_df, pivot_genreprice, pivot_genreprice_period)
    _df = add_pivots(_df, visit_pivot_genreprice, visit_pivot_genreprice_period)
   
    _df = add_past_purchase_key(_df)
    _df = add_pivots(_df, pivot_sarea, pivot_sarea_period)
    _df = add_pivots(_df, visit_pivot_sarea, visit_pivot_sarea_period)
    _df = feature_bin_values(_df, pivot_pref_distbin, pivot_pref, pivot_pref_period,
                              Utility.prefs, Utility.bin_names_buy)
    _df = feature_bin_values(_df, pivot_pref_distbin, visit_pivot_pref, visit_pivot_pref_period,
                              Utility.v_prefs, Utility.bin_names_visit)
    
    _df = _df.merge(coupon_pop , on=["COUPON_ID"], how="left")
    _df = _df.merge(coupon_areas_feature , on=["COUPON_ID", "user_pref"], how="left")
    _df["area"].fillna(0, inplace=True)
    
    _df["lnpop"] = np.log1p(_df["pop"])
    _df["lnDISCOUNT"] = np.log(_df["DISCOUNT_PRICE"] + 100.0)
    _df.fillna(0, inplace=True)
    
    _df["pref0"] = _df.dist == 0
    _df["pref24"] = _df.dist.isin([1,2,3])
    _df["spot"] = _df.genre.isin(Utility.spot_genre)
    _df["user_prefNN"] = _df.user_pref == "NN"

    pivot_sarea_columns       = pivot_sarea.columns.values
    visit_pivot_sarea_columns = visit_pivot_sarea.columns.values
    _df["pb_same_sarea"] = choose_column_by_keycol(_df, _df["small_area"], pivot_sarea_columns)
    _df["pb_same_v_sarea"] = choose_column_by_keycol(_df, "v_" + _df["small_area"], visit_pivot_sarea_columns)
    _df["pb_same_genre"] = choose_column_by_keycol(_df, _df["genre"], Utility.genres)
    _df["pb_same_v_genre"] = choose_column_by_keycol(_df, "v_" + _df["genre"], Utility.v_genres)
    _df["pb_same_pref"] = choose_column_by_keycol(_df, _df["pref"], Utility.prefs)
    _df["pb_same_v_pref"] = choose_column_by_keycol(_df, "v_" + _df["pref"], Utility.v_prefs)
    _df["pb_same_genreprice"] = choose_column_by_keycol(_df, _df["genreprice"], Utility.genreprices)
    _df["pb_same_v_genreprice"] = choose_column_by_keycol(_df, "v_" + _df["genreprice"], Utility.v_genreprices)

    LOG.info("create_dataframes_with_features finished")

    return _df

def feature_bin_values(_df, pref_bin_matrix_df, pivot, pivot_period, val_columns, bin_names):
    """
        Add sum of probability of purchase/visiting for each binned distance between pref and user_pref.
    """
    df = _df[["USER_ID", "period"]]
    df = add_pivots(df, pivot, pivot_period)

    val_matrix = df[val_columns].values 
    bin_matrix = pref_bin_matrix_df.reindex(_df.pref)[Utility.prefs].values # pref

    bv = bin_values(val_matrix, bin_matrix, Utility.bins)
    _df2 = pd.DataFrame(bv, index = _df.index, columns = bin_names)

    return pd.concat([_df, _df2], axis=1)

def bin_values(val_matrix, bin_matrix, bins):
    values = np.hstack([matrix_bin_sum(val_matrix, bin_matrix, b) for b in bins])
    return values

def matrix_bin_sum(val_matrix, bin_matrix, bin):
    return ((bin_matrix == bin) * val_matrix).sum(axis=1).reshape(-1, 1)


def add_pivots(_df, pivot, pivot_period):
    """
        If (USER_ID, period) matches pivot_period, get value of (USER_ID, period) from pivot_period.
        Else, get value of USER_ID from pivot.
    """

    idx1 = _df.USER_ID
    idx2 = zip(_df.USER_ID, _df.period)

    v1 = pivot.reindex(idx1).values
    v2 = pivot_period.reindex(idx2).values
    values = np.where(np.isnan(v2), v1, v2)       
    values = np.where(np.isnan(values), 0.0, values)

    _df2 = pd.DataFrame(values, index=_df.index, columns=pivot.columns)
    return pd.concat([_df, _df2], axis=1)

def add_past_purchase_key(_df):
    """
        Add purchased couponkey.
          - purchased couponkey is purchases of coupons with the same couponkey by the user.
          - exclude the information in the period where the coupon is.
    """
    key1 = Process.to_key1(_df)
    idx1 = zip(_df.USER_ID, key1)
    idx2 = zip(_df.USER_ID, key1, _df.period)

    pivot = past_buy_key
    pivot_period = past_buy_key_period

    v1 = pivot.reindex(idx1).values
    v1 = np.where(np.isnan(v1), 0, v1)  
    v2 = pivot_period.reindex(idx2).values
    v2 = np.where(np.isnan(v2), 0, v2)      
    values = v1 - v2

    _df2 = pd.DataFrame(values, index=_df.index, columns=["past_key"])
    return pd.concat([_df, _df2], axis=1)

def choose_column_by_keycol(_df, keycol, columns):
    """For col in columns, if col matches keycol value, get value of the column whose name is col."""

    ret = pd.Series(None, index=_df.index)
    for c in columns:
        ret[:] = np.where(keycol == c, _df[c], ret)
    ret.fillna(0, inplace=True)
    return ret
