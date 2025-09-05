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


from db.schema import Stats, Meas, DATE_FMT, DB_COLUMNS

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
    # NB: The DF_COLUMNS has two extra TSG sourced date columns that are not in the database schema
    # Remove to avoid confusion, drop TSG sourced columns as they will be merged in later
    new_df = new_df.drop(columns=['publish_date', 'hl_scan_date'])

    # Convert imported DataFrame column by column to usable data types
    for col in src_df.columns:
        # dates in db are converted to 'datetime.date' objects
        if col in ['modified_datetime']:
            new_df[col] = src_df[col].apply(conv_str2dt)
        # minerals, mineral counts and mineral data are converted to lists and dicts
        elif col in ['minerals', 'mincnts', 'data']:
            new_df[col] = src_df[col].apply(conv_str2json)
        # Don't merge in TSG sourced columns from src_df
        elif col in ['publish_date', 'hl_scan_date']:
            continue
        # Strings are left as is
        else:
            new_df[col] = src_df[col]

    # Merge in date columns from TSG files
    if not new_df.empty:
        merged_df = pd.merge(new_df, tsg_meta_df, left_on='nvcl_id', right_on='nvcl_id')
        # Rename from 'tsg_meta_df' column names to report column names
        merged_df = merged_df.rename(columns={'hl scan date': 'hl_scan_date', 'tsg publish date': 'publish_date'})
        return merged_df
    return pd.DataFrame(columns=DF_COLUMNS)


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
    Meas._meta.database = sdb
    models = generate_models(sdb)
    # If new database, create new table
    if 'meas' not in models:
        sdb.create_tables([Meas])
        models = generate_models(sdb)
    meas_mdl = models['meas']
    # Loop over all rows in dataframe to be exported
    for idx, row_arr in df.iterrows():

        # Assemble a dict from dataframe row in the dataframe
        row_df_dict = dict(zip(DF_COLUMNS, row_arr))
        # Insert missing report category
        row_df_dict['report_category'] = report_category
        # Convert Python data structures to strings
        row_df_dict['mincnts'] = conv_obj2str(row_df_dict['mincnts'])
        row_df_dict['minerals'] = conv_obj2str(row_df_dict['minerals'])
        row_df_dict['data'] = conv_obj2str(row_df_dict['data'])

        # Check date type
        if not isinstance(row_df_dict['modified_datetime'], date):
            print(f"ERROR: 'modified_datetime' in wrong format: {row_df_dict['modified_datetime']} in: {row_df_dict}")
            sys.exit(1)

        # Remove extra date columns sourced from TSG files
        del row_df_dict['publish_date']
        del row_df_dict['hl_scan_date']

        # Create new row in db
        try:
            tbl_handle = meas_mdl.create(**row_df_dict)
            tbl_handle.save()
        except peewee.IntegrityError as pie:
            # NB: Many rows with an 'empty' report_category will be duplicates
            print(f"Duplicate row error: {pie}")
            print("Tried to insert", row_df_dict)
        except peewee.InterfaceError as sie:
            print(f"Bad param error: {sie}")
            print("Tried to insert", row_df_dict)
            sys.exit(1)


def export_kms(db_file: str, prov_list: list, y: SimpleNamespace, q: SimpleNamespace):
    '''
    Export kms tables to db

    :param db_file: SQLITE database file name
    :param y: tuple of SimpleNamespace() fields are:
        start = start date
        end = end date
        kms_list = list of borehole kms in provider order
        cnts_list = list of borehole counts in provider order
    :param q: tuple of SimpleNamespace() fields are:
        start = start date
        end = end date
        kms_list = list of borehole kms in provider order
        cnts_list = list of borehole counts in provider order
    '''
    print(f"Opening: {db_file}")
    # Using peewee to write out rows
    sdb = SqliteDatabase(db_file)
    Stats._meta.database = sdb
    models = generate_models(sdb)
    # If new database, create new table
    if 'stats' not in models:
        sdb.create_tables([Stats])
    else:
        # Remove rows from table
        Stats.delete().execute()
    rows = []
    for idx, prov in enumerate(prov_list):
        rows.append({'stat_name': 'borehole_cnt_kms', 'provider': prov, 'start_date': y.start,
                    'end_date': y.end, 'stat_val1': y.cnt_list[idx], 'stat_val2': y.kms_list[idx]})
        rows.append({'stat_name': 'borehole_cnt_kms', 'provider': prov, 'start_date': q.start,
                    'end_date': q.end, 'stat_val1': q.cnt_list[idx], 'stat_val2': q.kms_list[idx]})
        
    try:
        # Try bulk insert
        print(f"Inserting {rows}")
        with sdb.atomic():
            Stats.insert_many(rows).execute()
    except peewee.IntegrityError as pie:
        print(f"Duplicate row error: {pie}")
        print("Tried to insert", rows)
        sys.exit(1)
    except peewee.InterfaceError as sie:
        print(f"Bad param error: {sie}")
        print("Tried to insert", rows)
        sys.exit(1)

