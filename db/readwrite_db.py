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
import math

import pandas
import numpy as np
import peewee
from peewee import SqliteDatabase, Model
from playhouse.reflection import generate_models


from db.schema import Meas, DATE_FMT, OLD_DATE_FMT

'''
Various routines used for reading and writing to the SQLITE db
'''

# Database columns = 'report_category' + list of columns in a new DataFrame
# List of columns in a new DataFrame is as follows:
DF_COLUMNS = ['provider', 'nvcl_id', 'modified_datetime', 'log_id', 'algorithm', 'log_type', 'algorithm_id', 'minerals', 'mincnts', 'data']



def df_col_str() -> str:
    '''
    Makes a comma sep string of dataframe column names
    '''
    return ', '.join(DF_COLUMNS)

#def db_col_str() -> str:
#    '''
#    Makes a comma sep string of database column names
#    '''
#    return 'report_category, ' + ', '.join(DF_COLUMNS)
#

def conv_dt(dt_str: str) -> datetime:
    '''
    Converts a date string in the YYYY/MM/DD format to a datetime.datetime object
    '''
    try:
        dt = datetime.strptime(dt_str, DATE_FMT)
    except ValueError:
        dt = datetime.strptime(dt_str, OLD_DATE_FMT)
    return dt

def conv_json(json_str):
    '''
    Converts from JSON string to Python object

    :param json_str: JSON string
    :returns: object or [] upon error
    '''
    if json_str == 'nan':
        return []
    try:
        return json.loads(json_str)
    except json.decoder.JSONDecodeError as jde:
        print(jde, "error decoding", repr(json_str))
        sys.exit(9)
    #    return []


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
    This exists because the older versions of NVCL Services do not return a date stamp.
    Gradually they will upgrade and the date stamps will be available.
    We have been inserting the time when the dataset was retrieved.
    Therefore this routine checks that the NVCL Services datetime is older and inserts it in the db

    :param con: SQLITE connection to db
    :param all_fields: tuple of all fields to be inserted into db
    '''
    #report_category, provider, nvcl_id, modified_datetime, log_id, algorithm, log_type, algorithm_id, minerals, mincnts, data  = all_fields
    #query = "select modified_datetime from meas where report_category=? and provider=? and nvcl_id=? and log_id=? and algorithm=? and log_type=? and algorithm_id=?"
    #cur = con.cursor()
    #index_tup = (report_category, provider, nvcl_id, log_id, algorithm, log_type, algorithm_id)
    #cur.execute(query, index_tup)
    #rows = cur.fetchall()

    # Get record from db
    db_rec = meas.select(meas.modified_datetime).where(
                       (meas.report_category == df_row['report_category']) &
                       (meas.provider == df_row['provider']) &
                       (meas.nvcl_id == df_row['nvcl_id']) &
                       (meas.log_id == df_row['log_id']) &
                       (meas.algorithm == df_row['algorithm']) &
                       (meas.log_type ==df_row['log_type']) &
                       (meas.algorithm_id == df_row['algorithm_id']))
    assert len(db_rec) == 1
    # If older than current value in db, then must be the more accurate NVCL Services, so update the row
    df_modified_dt = df_row['modified_datetime'].to_pydatetime().date()

    if datetime.strptime(db_rec[0].modified_datetime, DATE_FMT).date() > df_modified_dt:
        print("df_row=", df_row)
        print("db_rec=", repr(db_rec), db_rec, len(db_rec))
        print("md=", db_rec[0].modified_datetime)
        print(" !!! YES! ", repr(db_rec[0].modified_datetime), "should be replaced by", repr(df_modified_dt))
        db_rec[0].modified_datetime = df_modified_dt
        rec.save()
        print("REPLACED!")
        #cur.execute("update meas set modified_datetime = ? where report_category=? and provider=? and nvcl_id=? and log_id=? and algorithm=? and log_type=? and algorithm_id=?", (modified_datetime,) + index_tup)
    #else:
    #    print("NO! ", db_rec[0].modified_datetime, "not by replaced by", df_modified_dt)


def import_db(db_file: str, report_datacat: str):
    ''' 
    Reads a report category from SQLITE database converts it into a dataframe 
    Assumes file exists

    :param db_file: SQLITE database file name
    :param report_datacat: report category
    '''
    # Not using PeeWee, using Pandas to read db
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
    meas_mdl = models['meas']
    # Loop over all rows in dataframe
    for idx, row_arr in df.iterrows():
        # print(f"idx, row_arr.array={idx}, {row_arr.array}")
        row = dict(zip(DF_COLUMNS, row_arr))
        row['report_category'] = report_category
        if row['nvcl_id'] in known_ids:
            break
        try:
            # Create new row
            print('row=', repr(row))
            meas_mdl.create(**row)
            #Meas.create(report_category=report_category, provider=row['provider'],
            #          nvcl_id=row['nvcl_id'], modified_datetime=modified_datetime, log_id=row['log_id'],
            #          algorithm=row['algorithm'], log_type=row['log_type'],
            #          algorithmID=row['algorithm_id'], minerals=row['minerals'],
            #          mincnts=row['mincnts'], data=row['data'])
        except peewee.IntegrityError as pie:
            print("Duplicate row", pie)
            print("Tried to insert", row)
            # Update 'modified_datetime' if required
            update_datetime(meas_mdl, row)
        except peewee.InterfaceError as sie:
            print("Bad param", sie)
            print("Tried to insert", row)
            sys.exit(1)

if __name__ == "__main__":
    # Used for testing
    print(import_db("nvcl-test.db", "log2"))

