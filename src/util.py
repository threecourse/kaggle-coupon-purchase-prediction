"""
    This file is a utility module, having common fields and functions.
"""

import pandas as pd
import numpy as np
from util_logger import get_logger
LOG = get_logger()
LOG.info("load util")
        
class Utility:
    """Common fields"""

    spot_genre = ["Beauty", "Food", "Hair", "Health", "Hotel", "Leisure", "Nail",
                  "Relaxation", "Spa"]

    genres = list(pd.read_csv("../input/GENRE.csv")["en"])
    prefs = list(pd.read_csv("../input/PREF.csv")["en"])
    v_genres = ["v_" + g for g in genres]
    v_prefs = ["v_" + p for p in prefs]
    genreprices = ["Food1", "Food2", "Food9", "Hotel1", "Hotel2", "Hotel9", 
                   "Nail1", "Nail9", "Hair1", "Hair9", "Relaxation1", "Relaxation9"]
    v_genreprices = ["v_" + g for g in genreprices]

    bins = [0,1,2,3,4,5,6]
    bin_names_buy =   ["db_bin" + str(b) for b in bins]
    bin_names_visit = ["dv_bin" + str(b) for b in bins]

class Process:
    """Methods for processing"""

    @classmethod  
    def to_key1(cls, _df):
        """Create couponkey"""
        key1_columns = ["CATALOG_PRICE","DISCOUNT_PRICE","small_area","capsule"]
        return cls.str_series_cat([cls.to_str(_df[c]) for c in key1_columns])

    # private ------------------
    @classmethod  
    def str_series_cat(cls, lst_series):
        ret = lst_series[0]
        for series in lst_series[1:]:
           ret = ret + series
        return ret

    @classmethod  
    def to_str(cls, series):
        return series.astype(str)

class Grouping:
    """Methods for grouping"""
    
    @classmethod
    def to_group(cls, _df, columns, do_count):
       """Group by columns"""
       g = _df.groupby(columns).size().reset_index()
       g = g.rename(columns = {0:"count"})
       if not do_count : g = g.drop("count", axis = 1)
       return g

    @classmethod
    def to_group_count(cls, _df, columns):
       """Group by columns and sum up 'count' column"""
       g = _df.groupby(columns)["count"].sum().reset_index()
       return g

    @classmethod
    def to_pivotdf(cls, numer, denom, col):
        """
          Creates pivoted dataframe. 
            - Value is probability.
            - Rows are USER_IDs.
            - columns are elements of 'col' column in numer and denom.
        """
        cols1 = ["USER_ID"] + [col]

        numer2 = cls.to_group_count(numer, cols1)
        denom2 = cls.to_group_count(denom, cols1)
        prob = cls.prob(numer2, denom2, cols1)
        pivot = cls.pivot(prob,"USER_ID", col)
        return pivot 

    @classmethod
    def to_pivotdf_period(cls, numer, denom, col):
        """
          Creates pivoted dataframe.
            - Value is probability.
            - Rows are (USER_ID ,period) pairs.
            - Columns are elements of 'col' column in numer and denom.
        """
        cols1 = ["USER_ID"] + [col]

        numer_p = cls.to_group_count_period(numer, cols1)
        denom_p = cls.to_group_count_period(denom, cols1)
        numer_pall = cls.to_group_count(numer, cols1)
        denom_pall = cls.to_group_count(denom, cols1)

        prob = cls.prob_period(numer_p, denom_p, numer_pall, denom_pall, cols1)
        pivot = cls.pivot(prob, ["USER_ID", "period"], col)
        return pivot 

    @classmethod
    def lookup_coupon_element(cls, _df, coupon_df, col):
        """lookup column in coupons by COUPON_ID"""
        keys = _df.COUPON_ID.values
        se = pd.Series(coupon_df[col].values, index=coupon_df.COUPON_ID.values)
        return se[keys].values

    # private ---------------------------------

    @classmethod
    def to_group_count_period(cls, _df, columns):
       g2 = cls.to_group_count(_df, columns + ["period"])
       return g2

    @classmethod
    def prob(cls, numer, denom, columns):
        df = denom.merge(numer, on = columns, how="outer", suffixes=("", "_Y"))
        df["count"].fillna(0, inplace=True)
        df["count_Y"].fillna(0, inplace=True)
        df["count_N"] = df["count"] - df["count_Y"]
        df["prob"] = np.where(df["count"] == 0, 0.0, df["count_Y"] / df["count"])
        LOG.info("{} {} {}".format(len(df), df["count"].sum(), df["count_Y"].sum() ))
        return df

    @classmethod
    def prob_period(cls, numer_p, denom_p, numer_pall, denom_pall, columns):
        # specific period
        df = denom_p.merge(numer_p, on = columns + ["period"], how="outer", suffixes=("", "_Y"))
        df = df.rename(columns={"count":"count_p", "count_Y":"count_Y_p"})
        df["count_p"].fillna(0, inplace=True)
        df["count_Y_p"].fillna(0, inplace=True)
        
        # period all
        df = df.merge(denom_pall, on = columns, how="left").rename(columns={"count":"count_pall"})
        df = df.merge(numer_pall, on = columns, how="left").rename(columns={"count":"count_Y_pall"})
        df["count_pall"].fillna(0, inplace=True)
        df["count_Y_pall"].fillna(0, inplace=True)
        
        df["count"] = df["count_pall"] - df["count_p"]
        df["count_Y"] = df["count_Y_pall"] - df["count_Y_p"]

        df["count_N"] = df["count"] - df["count_Y"]
        df["prob"] = np.where(df["count"] == 0, 0.0, df["count_Y"] / df["count"])
        LOG.info("{} {} {}".format(len(df), df["count"].sum(), df["count_Y"].sum() ))
        return df

    @classmethod
    def pivot(cls, prob, index, column):
        df = prob.pivot_table(values="prob", index=index, columns=column)
        df.fillna(0, inplace=True)
        return df

    