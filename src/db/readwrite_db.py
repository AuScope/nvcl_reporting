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
from datetime import datetime, date
import json
import math

import pandas
import numpy as np
import peewee
from peewee import SqliteDatabase, Model
from playhouse.reflection import generate_models


from db.schema import Meas, DATE_FMT, DB_COLUMNS

# Dataframe columns = list of columns in database - 'report_category' + 'hl_scan_date' + 'publish_date'
from db.schema import DF_COLUMNS

'''
Various routines used for reading and writing to the SQLITE db

Unfortunately there is no automatic conversion between pandas dataframes and database fields with 
more complex types
i.e. pandas won't convert a column of arrays or json for your sqlite db even if you use sqlalchemy
'''

def db_col_str() -> str:
    '''
    Makes a comma sep string of column names for converting the database table to a DataFrame
    '''
    # Get dataframe columns
    db_cols = DF_COLUMNS.copy()
    # Remove columns sourced from TSG files
    db_cols.remove('hl_scan_date')
    db_cols.remove('publish_date')
    return ', '.join(db_cols)


def conv_str2dt(dt_str: str) -> date:
    '''
    Converts a date string in the YYYY-MM-DD format to a datetime.date object
    '''
    #assert isinstance(dt_str, str) and dt_str[4] == '/' and dt_str[7] == '/' and dt_str[:3].isnumeric()
    assert type(dt_str) is not pd.Timestamp

    try:
        return datetime.strptime(dt_str, DATE_FMT).date()
    except ValueError:
        dt = datetime.now().date()
        print("dt_str", repr(dt_str), "not a valid string")
        return dt


def conv_str2json(json_str) -> list:
    '''
    Converts from JSON string to Python list object

    :param json_str: JSON string
    :returns: object or [] upon error
    '''
    if json_str == 'nan':
        return []
    try:
        return json.loads(json_str)
    except json.decoder.JSONDecodeError as jde:
        print(jde, "error decoding", repr(json_str))
        sys.exit(1)


#def to_date(timestamp: pd.Timestamp) -> str:
#    '''
#    Converts Pandas Timestamp to string
#
#    :param timestamp: Timestamp 
#    :return: string
#    '''
#    dt = timestamp.to_pydatetime()
#    return dt.strftime(DATE_FMT)


def import_db(db_file: str, report_datacat: str, tsg_meta_df: pd.DataFrame) -> pd.DataFrame:
    ''' 
    Reads a report category from SQLITE database converts it into a dataframe 
    Assumes file exists

    :param db_file: SQLITE database file name
    :param report_datacat: report category
    :returns: new pandas dataframe
    '''

    # Read category data from database
    try:
        con = sqlite3.connect(db_file)
    except sqlite3.Error as s3e:
        print(f"Cannot connect to database {db_file}: {s3e}")
        sys.exit(1)

    # Convert db data to dataframe
    try:
        src_df = pd.read_sql(f"select {db_col_str()} from meas where report_category = '{report_datacat}'", con)
    except pandas.io.sql.DatabaseError as de:
        # If does not exist, create a new one
        print(f"Cannot find data in database {db_file}: {de}")
        print("Creating a new database")
        src_df = pd.DataFrame(columns=DF_COLUMNS)
    assert type(src_df['modified_datetime']) is not pd.Timestamp

    # Create new frame for populating
    new_df = pd.DataFrame(columns=DF_COLUMNS)
    # Remove to avoid confusion, as they will be merged in
    new_df = new_df.drop(columns=['publish_date', 'hl_scan_date'])

    # Convert imported DataFrame column by column to usable data types
    for col in src_df.columns:
        # dates in db are converted to 'datetime.date' objects
        if col in ['modified_datetime']:
            new_df[col] = src_df[col].apply(conv_str2dt)
        # minerals, mineral counts and mineral data are converted to lists and dicts
        elif col in ['minerals', 'mincnts', 'data']:
            new_df[col] = src_df[col].apply(conv_str2json)
        # Strings are left as is
        else:
            new_df[col] = src_df[col]

    # Merge columns from TSG files
    merged_df = pd.merge(new_df, tsg_meta_df, left_on='nvcl_id', right_on='nvcl_id')
    # Rename from 'tsg_meta_df' column names to report column names
    merged_df = merged_df.rename(columns={'hl scan date': 'hl_scan_date', 'tsg publish date': 'publish_date'})
    return merged_df


def conv_obj2str(arr: []) -> str:
    """ Converts simple list object to JSON-style string """
    try:
        return json.dumps(arr)
    except json.decoder.JSONDecodeError as jde:
        print(f"{jde}: error decoding {arr}")
        sys.exit(9)


def export_db(db_file: str, df: pd.DataFrame, report_category: str, tsg_meta_df: pd.DataFrame):
    '''
    Writes entire dataframe to SQLITE database

    :param db_file: SQLITE database file name
    :param df: dataframe whose rows are exported to db
    :param report_category: report category
    :param tsg_meta_df: TSG metadata dataframe
    '''
    # DB cols: report_category, provider, nvcl_id, modified_datetime, log_id, algorithm, log_type, algorithm_id, minerals, mincnts, data 
    #
    # Using peewee to write out rows
    sdb = SqliteDatabase(db_file)
    models = generate_models(sdb)
    # If new database, create new table
    if 'meas' not in models:
        Meas._meta.database = sdb
        sdb.create_tables([Meas])
        models = generate_models(sdb)
    meas_mdl = models['meas']
    # Loop over all rows in dataframe to be exported
    for idx, row_arr in df.iterrows():

        # Assemble a dict from dataframe row
        row_df_dict = dict(zip(DB_COLUMNS, row_arr))
        row_df_dict['report_category'] = report_category
        # Convert Python data structures to strings
        row_df_dict['mincnts'] = conv_obj2str(row_df_dict['mincnts'])
        row_df_dict['minerals'] = conv_obj2str(row_df_dict['minerals'])
        row_df_dict['data'] = conv_obj2str(row_df_dict['data'])

        ## Check data type
        assert isinstance(row_df_dict['modified_datetime'], date) 
        assert isinstance(row_df_dict['hl_scan_date'], date)

        # Create new row in db
        try:
            tbl_handle = meas_mdl.create(**row_df_dict)
            tbl_handle.save()
        except peewee.IntegrityError as pie:
            # NB: Many rows with an 'empty' report_category will be duplicates
            print("Duplicate row", pie)
            print("Tried to insert", row_df_dict)
        except peewee.InterfaceError as sie:
            print("Bad param", sie)
            print("Tried to insert", row_df_dict)
            sys.exit(1)
