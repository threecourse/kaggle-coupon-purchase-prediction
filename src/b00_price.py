"""
    This file creates genre-price and standarized log price in the genre.
    Genre-price is genre divided by price range.
"""

import pandas as pd
import numpy as np
from util_logger import get_logger
LOG = get_logger()
LOG.info("start b00")

# load files
coupons = pd.read_pickle("../model/coupons.pkl")

def get_genre_price(genre, price):
    """Return genre-price from genre and price range"""
    
    if genre == "Food" and price <= 1500 : return "Food1"
    if genre == "Food" and price <= 3000 : return "Food2"
    if genre == "Food"                   : return "Food9"
    if genre == "Hotel" and price <= 5000 : return "Hotel1"
    if genre == "Hotel" and price <= 10000 : return "Hotel2"
    if genre == "Hotel"                   : return "Hotel9"
    if genre == "Nail" and price <= 3500 : return "Nail1"
    if genre == "Nail"                   : return "Nail9"
    if genre == "Hair" and price <= 4000 : return "Hair1"
    if genre == "Hair"                   : return "Hair9"
    if genre == "Relaxation" and price <= 3000 : return "Relaxation1"
    if genre == "Relaxation"                   : return "Relaxation9"
    return "NNgp"

# add genre-price
coupons["genreprice"] = map(get_genre_price, coupons.genre, coupons.DISCOUNT_PRICE)

# calculate standarized log price in the genre
coupons["lnDPRICE"] = np.log1p(coupons["DISCOUNT_PRICE"])
coupons["mDPRICE"] = coupons.groupby("genre")["lnDPRICE"].transform(np.mean)
coupons["sDPRICE"] = coupons.groupby("genre")["lnDPRICE"].transform(np.std)
coupons["zprice"] = (coupons["lnDPRICE"] - coupons["mDPRICE"]) / coupons["sDPRICE"]

# write
coupons_price = coupons[["COUPON_ID", "genreprice", "zprice"]]
coupons_price.to_pickle("../model/coupons_price.pkl")

LOG.info("finished")