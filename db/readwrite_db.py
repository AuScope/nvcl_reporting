#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import pandas as pd
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', None)
import sqlite3
from collections import OrderedDict
from collections.abc import Iterable
from types import SimpleNamespace
from datetime import datetime
import json
import pandas
import numpy as np


'''
Various routines used for reading and writing to the SQLITE db
'''

# Database columns = 'report_category' + list of columns in a new DataFrame
# List of columns in a new DataFrame is as follows:
DF_COLUMNS = ['provider', 'nvcl_id', 'modified_datetime', 'log_id', 'algorithm', 'log_type', 'algorithm_id', 'minerals', 'mincnts', 'data']

# Dates are stored as strings in this format
DATE_FORMAT = '%Y/%m/%d'


def df_col_str() -> str:
    '''
    Makes a comma sep string of dataframe column names
    '''
    return ', '.join(DF_COLUMNS)

def db_col_str() -> str:
    '''
    Makes a comma sep string of database column names
    '''
    return 'report_category, ' + ', '.join(DF_COLUMNS)

def conv_dt(dt_str) -> datetime:
    '''
    Converts a date string in the YYYY/MM/DD format to a datetime.datetime object
    '''
    return datetime.strptime(dt_str, DATE_FORMAT)

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


def conv_minerals(minerals) -> str:
    '''
    Converts lists of minerals to JSON formatted string

    :param minerals: list of minerals
    :returns: JSON formatted string
    '''
    # Sometimes 'minerals' is not an iterable numpy array
    if isinstance(minerals, Iterable):
        minerals_out = json.dumps(list(minerals))
    elif isinstance(minerals, float) and math.isnan(minerals):
        minerals_out = []
    else:
        minerals_out = json.dumps([minerals])
    return minerals_out


def conv_mincnts(mincnts) -> str:
    '''
    Converts lists of minerals counts to JSON formatted string

    :param minerals: list of mineral counts
    :returns: JSON formatted string
    '''
    # Sometimes 'mincnts' is not an iterable numpy array
    if isinstance(mincnts, Iterable):
        # convert numpy int64 -> int
        mincnts_out = json.dumps([int(cnt) for cnt in mincnts])
    elif isinstance(mincnts, float) and math.isnan(mincnts):
        mincnts_out = []
    else:
        mincnts_out = json.dumps([mincnts])
    return mincnts_out


def conv_data(data) -> str:
    '''
    Convert mineral types at each depth to JSON formatted string
    i.e. ((depth, {key: val ...}, ...) ... ) or NaN

    :param data: mineral types at each depth
    :returns: JSON formatted string
    '''
    #print(f"data={data}")
    data_list = []
    if isinstance(data, OrderedDict):
        for depth, obj in data.items():
            # vars() converts Namespace -> dict
            if isinstance(obj, SimpleNamespace):
                data_list.append([depth, vars(obj)])
            elif isinstance(obj, list) and len(obj) == 0:
                continue
            else:
                print(repr(obj), type(obj))
                print("ERROR Unknown obj type in 'data' var")
                sys.exit(1)
    elif data != {} and data != [] and not (isinstance(data, float) and math.isnan(data)):
        print(repr(data), type(data))
        print("ERROR Unknown type in 'data' var")
        sys.exit(1)

    return json.dumps(data_list)


def to_date(timestamp: pd.Timestamp) -> str:
    '''
    Converts Pandas Timestamp to string

    :param timestamp: Timestamp 
    :return: string
    '''
    dt = timestamp.to_pydatetime()
    return dt.strftime(DATE_FORMAT)


def import_db(db_file, report_datacat):
    ''' 
    Reads a report category from SQLITE database converts it into a dataframe 
    Assumes file exists

    :param db_file: SQLITE database file name
    :param report_datacat: report category
    '''
    con = sqlite3.connect(db_file)
    try:
        df = pd.read_sql(f"select {df_col_str()} from meas where report_category = '{report_datacat}'", con) 
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


def export_db(db_file, df, report_category):
    '''
    Writes a dataframe to SQLITE database

    :param db_file: SQLITE database file name
    :param df: dataframe
    :param report_category: report category
    '''
    con = sqlite3.connect(db_file)
    curs = con.cursor()
    for idx, row in df.iterrows():
        # print(f"idx, row.array={idx}, {row.array}")
        # report_category, provider, nvcl_id, modified_datetime, log_id, algorithm, log_type, algorithmID, minerals, mincnts, data 
        qu_str = '?,' + ','.join([ '?' for i in DF_COLUMNS ])
        insert_str = f"INSERT INTO MEAS({db_col_str()}) VALUES({qu_str});"
        row_dict = dict(zip(DF_COLUMNS, row))
        insert_tup = (report_category,)
        for key in DF_COLUMNS:
            # print(key, type(row_dict[key]))
            # Timestamp
            if isinstance(row_dict[key], pd.Timestamp):
                insert_tup += (to_date(row_dict[key]),)
            # Data
            elif key == 'data':
                insert_tup += (conv_data(row_dict[key]),)
            # Mincnts
            elif key == 'mincnts':
                insert_tup += (conv_mincnts(row_dict[key]),)
            # Minerals
            elif key == 'minerals':
                insert_tup += (conv_minerals(row_dict[key]),)
            # String
            else:
                insert_tup += (row_dict[key],)

        curs.execute(insert_str, insert_tup)
    con.commit()
    #assert(num_rows == len(df.index))


if __name__ == "__main__":
    print(import_db("nvcl-test.db", "log2"))

