#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import pandas as pd
pd.options.mode.chained_assignment = None
import sqlite3
from collections import OrderedDict
from types import SimpleNamespace


def get_data(data_str, col_name):
    if col_name == 'data':
        return eval(data_str.replace('namespace', 'SimpleNamespace'))
    return(eval(data_str))

if __name__ == "__main__":

    # Open up connection
    #db.connect()
    #records = Meas.get()
    #print("resords=", records)
    con = sqlite3.connect("nvcl2.db")
    df = pd.read_sql("SELECT * from MEAS", con) 
    for col in ['report_category', 'state', 'nvcl_id', 'create_datetime', 'log_id', 'algorithm', 'log_type', 'algorithmID', 'minerals', 'mincnts', 'data']:
        print(repr(get_data(df[col][0], col)), type(get_data(df[col][0], col)))
    #for v in get_data(df['data'][6]).values():
    #    print(v.classText)
