#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import pandas as pd
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', None)
import sqlite3
from collections import OrderedDict
from types import SimpleNamespace
from datetime import datetime
from numpy import array
import json



def conv_dt(dt_str):
    return datetime.strptime(dt_str, '%Y/%m/%d')

def conv_json(json_str):
    return json.loads(json_str) 

def import_db(db_file, report_datacat):
    if not os.path.exists(db_file):
        print(f"{db_file} does not exist")
        sys.exit(1)
    con = sqlite3.connect(db_file)
    df = pd.read_sql(f"select provider, nvcl_id, create_datetime, log_id, algorithm, log_type, algorithmID, minerals, mincnts, data from meas where report_category = '{report_datacat}'", con) 
    new_df = pd.DataFrame()
    for col in df.columns:
        #print(f"converting {col}")
        if col == 'create_datetime':
            new_df[col] = df[col].apply(conv_dt)
        elif col in ['minerals', 'mincnts', 'data']:
            new_df[col] = df[col].apply(conv_json)
        else:
            new_df[col] = df[col]

    return new_df


if __name__ == "__main__":
    print(import_db("nvcl-test.db", "log2"))

