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
import pandas



def conv_dt(dt_str):
    return datetime.strptime(dt_str, '%Y/%m/%d')

def conv_json(json_str):
    ''' 
    Converts from JSON string to Python object

    :param json_str: JSON string
    :returns: object or [] upon error
    '''
    try:
        return json.loads(json_str) 
    except json.decoder.JSONDecodeError:
        return []

def import_db(db_file, report_datacat):
    ''' 
    Reads a report category from database converts it into a dataframe 
    Assumes file exists

    :param db_file: SQLITE database file name
    :param report_datacat: report category
    '''
    con = sqlite3.connect(db_file)
    try:
        df = pd.read_sql(f"select provider, nvcl_id, modified_datetime, log_id, algorithm, log_type, algorithmID, minerals, mincnts, data from meas where report_category = '{report_datacat}'", con) 
    except pandas.io.sql.DatabaseError as de:
        print(f"Cannot find data category in database {db_file}: {de}")
        exit(1)
    new_df = pd.DataFrame()
    for col in df.columns:
        #print(f"converting {col}")
        if col == 'modified_datetime':
            new_df[col] = df[col].apply(conv_dt)
        elif col in ['minerals', 'mincnts', 'data']:
            new_df[col] = df[col].apply(conv_json)
        else:
            new_df[col] = df[col]

    return new_df


def export_db(db_file, df):
    '''
    Writes a dataframe to SQLITE database

    :param db_file: SQLITE database file name
    :param df: dataframe
    '''
    con = sqlite3.connect(db_file)
    try:
        num_rows = df.to_sql('meas', con, if_exists='append', index=False)
    except ValueError as ve:
        print(f"Cannot insert values into database {db_file}: {ve}")
        sys.exit(1)
    assert(num_rows == len(df.index))


if __name__ == "__main__":
    print(import_db("nvcl-test.db", "log2"))

