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


from db.schema import Meas, DATE_FMT, DF_COLUMNS
from db.tsg_metadata import TSGMeta

# Database columns = 'report_category' + list of columns in a new DataFrame
from db.schema import DF_COLUMNS

'''
Various routines used for reading and writing to the SQLITE db

Unfortunately there is no automatic conversion between pandas dataframes and database fields with 
more complex types
i.e. pandas won't convert a column of arrays or json for your sqlite db even if you use sqlalchemy
'''

def df_col_str() -> str:
    '''
    Makes a comma sep string of dataframe column names
    '''
    return ', '.join(DF_COLUMNS)


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


def import_db(db_file: str, report_datacat: str) -> pd.DataFrame:
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
        df = pd.read_sql(f"select {df_col_str()} from meas where report_category = '{report_datacat}'", con) 
    except pandas.io.sql.DatabaseError as de:
        # If does not exist, create a new one
        print(f"Cannot find data in database {db_file}: {de}")
        print("Creating a new database")
        df = pd.DataFrame(columns=DF_COLUMNS)
    assert type(df['modified_datetime']) is not pd.Timestamp

    # Create new frame for populating
    new_df = pd.DataFrame(columns=DF_COLUMNS)

    # Convert imported DataFrame column by column to usable data types
    for col in df.columns:
        # dates in db are converted to 'datetime.date' objects
        if col in ['modified_datetime','hl_scan_date']:
            new_df[col] = df[col].apply(conv_str2dt)
        # minerals, mineral counts and mineral data are converted to lists and dicts
        elif col in ['minerals', 'mincnts', 'data']:
            new_df[col] = df[col].apply(conv_str2json)
        # Strings are left alone
        else:
            new_df[col] = df[col]

    assert type(new_df['modified_datetime']) is not pd.Timestamp
    assert type(new_df['hl_scan_date']) is not pd.Timestamp
    return new_df


def conv_obj2str(arr: []) -> str:
    """ Converts simple list object to JSON-style string """
    try:
        return json.dumps(arr)
    except json.decoder.JSONDecodeError as jde:
        print(f"{jde}: error decoding {arr}")
        sys.exit(9)


def export_db(db_file: str, df: pd.DataFrame, report_category: str, tsg_meta: TSGMeta):
    '''
    Writes entire dataframe to SQLITE database

    :param db_file: SQLITE database file name
    :param df: dataframe whose rows are exported to db
    :param report_category: report category
    :param tsg_meta: TSG metadata used to fetch Hylogger scan date
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
        row_df_dict = dict(zip(DF_COLUMNS, row_arr))
        row_df_dict['report_category'] = report_category
        row_df_dict['mincnts'] = conv_obj2str(row_df_dict['mincnts'])
        row_df_dict['minerals'] = conv_obj2str(row_df_dict['minerals'])
        row_df_dict['data'] = conv_obj2str(row_df_dict['data'])

        ## Check data type
        assert isinstance(row_df_dict['modified_datetime'], date) 
        assert isinstance(row_df_dict['hl_scan_date'], date)

        # Assuming incremental updates, update 'hl_scan_date'
        # The 'hl_scan_date' comes from a source that may be updated after the NVCLDataServices
        # so initially 'hl_scan_date' maybe unavailable
        hl_scan_date = tsg_meta.get_hl_scan_date(row_df_dict['nvcl_id'])
        if hl_scan_date is not None:
            print(f"Changing {row_df_dict['hl_scan_date']} to {hl_scan_date}")
            row_df_dict['hl_scan_date'] = hl_scan_date

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
