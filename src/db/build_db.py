#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
from pathlib import Path
import datetime
from datetime import date
import time
import pickle
import sqlite3
import contextlib
from collections.abc import Iterable
from collections import OrderedDict
import json
from types import SimpleNamespace
import math

import pandas as pd
pd.options.mode.chained_assignment = None

from peewee import SqliteDatabase, Model, TextField, DateField, CompositeKey, IntegrityError

from db.schema import DATE_FMT, Meas

DB_NAME = 'nvcl-test.db'
if os.path.exists(DB_NAME):
    os.unlink(DB_NAME)
db = SqliteDatabase(DB_NAME)


def import_pkl(infile: str, empty={}) -> any:
    """ Reads pickle file and returns its data
    NB: exits upon exception

    :param infile: filename of pickle file
    :param empty: optional returns this if pickle file not found
    :returns: pickle file data or 'empty' or {} if empty not defined
    """
    print(f"Importing {infile} ...")
    if not os.path.exists(infile):
        print(f"{infile} not found, assuming it is empty")
        return empty
    try:
        data = pd.read_pickle(infile)
    except ValueError:
        try:
            with open(infile, 'rb') as handle:
                data = pickle.load(handle)
        except pickle.UnpicklingError as pe:
            print(f"Could not load pickle {infile}: {pe}")
            sys.exit(1)
    except Exception as exc:
        print(f"Error reading pickle file {infile}: {exc}")
        sys.exit(1)
    return data

def convert2json(modified_dt: datetime, minerals: [], mincnts: [], data: any) -> (str, str, str, str):
    """ Converts fields to JSON strings in preparation for db insert

    :param modified_st: record's modified datetime
    :param mineral: list of mineral names
    :param mincnts: mineral counts
    :param data: mineral class at depths data
    :returns: a tuple of JSON strings
    """

    modified_dt_out = modified_dt.strftime(DATE_FMT)
    # Sometimes 'minerals' is not an iterable numpy array
    if isinstance(minerals, Iterable):
        minerals_out = json.dumps(list(minerals))
    elif isinstance(minerals, float) and math.isnan(minerals):
        minerals_out = []
    else:
        minerals_out = json.dumps([minerals])
    # Sometimes 'mincnts' is not an iterable numpy array
    if isinstance(mincnts, Iterable):
        # convert numpy int64 -> int
        mincnts_out = json.dumps([int(cnt) for cnt in mincnts])
    elif isinstance(mincnts, float) and math.isnan(mincnts):
        mincnts_out = []
    else:
        mincnts_out = json.dumps([mincnts])
    # Convert 'data' i.e. ((depth, {key: val ...}, ...) ... ) or NaN
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
                print("ERROR unknown obj type in 'data' var")
                sys.exit(1)
    elif data != {} and data != [] and not (isinstance(data, float) and math.isnan(data)):
        print(repr(data), type(data))
        print("ERROR unknown type in 'data' var")
        sys.exit(1)

    data_out = json.dumps(data_list)

    return modified_dt_out, minerals_out, mincnts_out, data_out


def load_data(pickle_dir: str, subdir: str):
    """ Load NVCL data from pickle file

    :param pickle_dir: directory path from which to load pickle file
    :param subdir: source subdirectory

    report_category = TextField()
    provider = TextField()
    nvcl_id = TextField()
    modified_datetime = DateField()
    log_id = TextField()
    algorithm = TextField()
    log_type = TextField()
    algorithm_id = TextField()
    minerals = TextField()
    mincnts = TextField()
    data = TextField()
    """
    # Translate sub dirs to report categories
    if subdir == 'other':
        report_category = 'log2'
    elif subdir == 'data':
        report_category = 'log1'
    else:
        report_category = subdir

    # Read all pickle files in dir
    dir_path = Path(pickle_dir)
    for pickle_file in [pf for pf in dir_path.iterdir() if pf.is_file()]:
        df = import_pkl(pickle_file)
        print(f"{pickle_file}:")
        print(repr(df))
        print(df[['state', 'nvcl_id', 'log_id', 'algorithm']].head(10))
        # Read file timestamp, to be used as modified_datetime
        # Modified date is not stored in the pickle files, so have to use the oldest file timestamp
        # from all the pickle files as an approximation
        try:
            modified_dt = datetime.datetime.fromtimestamp(pickle_file.stat().st_mtime)
        except Exception:
            print(f"Could not find date for {pickle_file}")
        else:
            print(f"Modified date for pickle file is {modified_dt}")
        # Loop over the rows in the pandas DataFrame, converting fields to strings or JSON
        for idx, row in df.iterrows():
            # NB: 'metres' is better labelled as 'mincnts'
            modified_datetime, minerals, mincnts, data = convert2json(modified_dt, row['minerals'],
                                                                    row['metres'], row['data'])
            #print("Inserting", row['state'], row['nvcl_id'], row['log_id'], row['algorithm'])
            try:
                Meas.create(report_category=report_category, provider=row['state'],
                      nvcl_id=row['nvcl_id'], modified_datetime=modified_datetime, log_id=row['log_id'],
                      algorithm=row['algorithm'], log_type=row['log_type'],
                      algorithm_id=row['algorithm_id'], minerals=minerals,
                      mincnts=mincnts, data=data)
            except IntegrityError:
                # This is a duplicate, so modify modification date if earlier than the stored one
                rec = Meas.get(report_category=report_category, provider=row['state'],
                       nvcl_id=row['nvcl_id'], log_id=row['log_id'], algorithm=row['algorithm'],
                       log_type=row['log_type'], algorithm_id=row['algorithm_id'])
                rec_modified_dt = modified_dt.strptime(rec.modified_datetime, DATE_FMT)
                # Alter modified date if earlier
                if modified_dt < rec_modified_dt:
                    rec.modified_datetime = modified_dt.strftime(DATE_FMT)
                    rec.save()
        print("Finished inserting")


if __name__ == "__main__":

    # Directory where all NVCL data pickle files exist
    pickle_dir = 'all_pkl_dir'
    pkl_path = Path(pickle_dir)
    if not pkl_path.exists():
        os.mkdir(pickle_dir)

    # Open up connection
    db.connect()
    db.create_tables([Meas])

    # Open up NVCL data pickle files, extract modified dates and create a new sqlite db
    for subdir in ['data','other','emptyrecs','nodata']:
        pkl_subdir = Path(pickle_dir) / Path(subdir)
        print("Loading", pkl_subdir)
        load_data(pkl_subdir, subdir)
    print("Done.")
