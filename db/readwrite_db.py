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


def update_datetime(meas: Model, df_row: dict):
    '''
    If the record already exists in the database, then check if modified_datetime needs updating.
    If so, then update the database.

    This exists because the older versions of NVCL Services do not return a date stamp.
    Gradually they will upgrade and the date stamps will be available.
    We have been inserting the time when the dataset was retrieved.
    Therefore this routine checks that the NVCL Services datetime is older and inserts it in the db

    :param con: SQLITE connection to db
    :param all_fields: tuple of all fields to be inserted into db
    '''
    # Get record from db
    db_rec = meas.select().where(
                       (meas.report_category == df_row['report_category']) &
                       (meas.provider == df_row['provider']) &
                       (meas.nvcl_id == df_row['nvcl_id']) &
                       (meas.log_id == df_row['log_id']) &
                       (meas.algorithm == df_row['algorithm']) &
                       (meas.log_type ==df_row['log_type']) &
                       (meas.algorithm_id == df_row['algorithm_id']))
    assert len(db_rec) == 1
    df_modified_dt = df_row['modified_datetime']
    assert type(df_modified_dt) is not pd.Timestamp

    db_modified_dt = db_rec[0].modified_datetime
    assert type(db_modified_dt) is not pd.Timestamp
    #print(f"{df_modified_dt=}")
    #print(f"{db_modified_dt=}")

    # If older than current value in db, then must be the more accurate NVCL Services, so update the row
    if db_modified_dt > df_modified_dt:
        db_rec[0].modified_datetime = df_modified_dt
        db_rec[0].save()


def import_db(db_file: str, report_datacat: str) -> pd.DataFrame:
    ''' 
    Reads a report category from SQLITE database converts it into a dataframe 
    Assumes file exists

    :param db_file: SQLITE database file name
    :param report_datacat: report category
    :returns: new pandas dataframe
    '''

    # Read category data from database
    con = sqlite3.connect(db_file)
    try:
        df = pd.read_sql(f"select {df_col_str()} from meas where report_category = '{report_datacat}'", con) 
    except pandas.io.sql.DatabaseError as de:
        # If does not exist, create a new one
        print(f"Cannot find data in database {db_file}: {de}")
        print("Creating a new database")
        df = pd.DataFrame(columns=DF_COLUMNS)
    assert type(df['modified_datetime']) is not pd.Timestamp

    # Convert to usable datatypes
    new_df = pd.DataFrame(columns=DF_COLUMNS)
    for col in df.columns:
        #print(f"converting {col}")
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
    return new_df


def conv_obj2str(arr: []) -> str:
    """ Converts simple list object to JSON-style string """
    try:
        return json.dumps(arr)
    except json.decoder.JSONDecodeError as jde:
        print(f"{jde}: error decoding {arr}")
        sys.exit(9)


def export_db(db_file: str, df: pd.DataFrame, report_category: str, known_ids: []):
    '''
    Writes a dataframe to SQLITE database

    :param db_file: SQLITE database file name
    :param df: dataframe
    :param report_category: report category
    :param known_ids: list of NVCL ids that are already in the database
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
    # Loop over all rows in dataframe
    for idx, row_arr in df.iterrows():
        #print(f"{DF_COLUMNS=}")
        #print(f"{[cols for cols in df.columns]}")
        #print(f"{df.head()=}")

        # Assemble a dict from dataframe row
        row_df_dict = dict(zip(DF_COLUMNS, row_arr))
        row_df_dict['report_category'] = report_category
        row_df_dict['mincnts'] = conv_obj2str(row_df_dict['mincnts'])
        row_df_dict['minerals'] = conv_obj2str(row_df_dict['minerals'])
        row_df_dict['data'] = conv_obj2str(row_df_dict['data'])

        assert type(row_df_dict['modified_datetime']) is not pd.Timestamp

        
        # !!! FIXME: Temporary
        #if row_df_dict['nvcl_id'] in known_ids:
        #    break
        try:
            # Create new row in db
            #print(f"{row_df_dict=}")
            tbl_handle = meas_mdl.create(**row_df_dict)
            tbl_handle.save()
        except peewee.IntegrityError as pie:
            print("Duplicate row", pie)
            print("Tried to insert", row_df_dict)
            # Update 'modified_datetime' if required
            update_datetime(meas_mdl, row_df_dict)
        except peewee.InterfaceError as sie:
            print("Bad param", sie)
            print("Tried to insert", row_df_dict)
            sys.exit(1)

if __name__ == "__main__":
    # Used for testing
    print(import_db("nvcl-test.db", "log2"))

