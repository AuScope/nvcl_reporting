#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
from pathlib import Path
import pandas as pd
pd.options.mode.chained_assignment = None
import datetime
import time
import pickle
import sqlite3
import contextlib
from collections.abc import Iterable
from collections import OrderedDict
import json
from types import SimpleNamespace
import math

from peewee import SqliteDatabase, Model, TextField, DateField, CompositeKey, IntegrityError
from datetime import date

DATE_FMT = '%Y/%m/%d'
DB_NAME = 'nvcl2.db'
if os.path.exists(DB_NAME):
    os.unlink(DB_NAME)
db = SqliteDatabase(DB_NAME)

class Meas(Model):
    # ['state', 'nvcl_id', 'create_datetime', 'log_id', 'algorithm', 'log_type', 'algorithmID', 'minerals', 'metres', 'data']
    report_category = TextField() # Can be any one of 'log1', 'log2', 'empty' and 'nodata'
    state = TextField()
    nvcl_id = TextField()
    create_datetime = DateField()
    log_id = TextField()
    algorithm = TextField()
    log_type = TextField()
    algorithmID = TextField()
    minerals = TextField() # Unique minerals
    mincnts = TextField()  # Counts of unique minerals as an array
    data = TextField()     # Raw data as a dict

    class Meta:
        primary_key = CompositeKey('report_category', 'state', 'nvcl_id', 'log_id', 'algorithm', 'log_type', 'algorithmID')
        database = db


def import_pkl(infile, empty={}):
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

def convert2json(create_dt, minerals, mincnts, data):

    create_dt_out = create_dt.strftime(DATE_FMT)
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

    return create_dt_out, minerals_out, mincnts_out, data_out

def load_data(pickle_dir, subdir):
    """ Load NVCL data from pickle file

    :param pickle_dir: directory path from which to load pickle file
    :param subdir: source subdirectory

    report_category = TextField()
    state = TextField()
    nvcl_id = TextField()
    create_datetime = DateField()
    log_id = TextField()
    algorithm = TextField()
    log_type = TextField()
    algorithmID = TextField()
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
        # Read file timestamp, to be used as create_date
        # Create date is not stored in the pickle files, so have to use the oldest file timestamp
        # from all the pickle files as an approximation
        try:
            create_dt = datetime.datetime.fromtimestamp(pickle_file.stat().st_mtime)
        except Exception:
            print(f"Could not find date for {pickle_file}")
        else:
            print(f"Create date for pickle file is {create_dt}")
        # Loop over the rows in the pandas DataFrame, converting fields to strings or JSON
        for idx, row in df.iterrows():
            # NB: 'metres' is better labelled as 'mincnts'
            create_datetime, minerals, mincnts, data = convert2json(create_dt, row['minerals'],
                                                                    row['metres'], row['data'])
            #print("Inserting", row['state'], row['nvcl_id'], row['log_id'], row['algorithm'])
            try:
                Meas.create(report_category=report_category, state=row['state'],
                      nvcl_id=row['nvcl_id'], create_datetime=create_datetime, log_id=row['log_id'],
                      algorithm=row['algorithm'], log_type=row['log_type'],
                      algorithmID=row['algorithmID'], minerals=minerals,
                      mincnts=mincnts, data=data)
            except IntegrityError:
                # This is a duplicate, so modify create date if earlier than the stored one
                rec = Meas.get(report_category=report_category, state=row['state'],
                       nvcl_id=row['nvcl_id'], log_id=row['log_id'], algorithm=row['algorithm'],
                       log_type=row['log_type'], algorithmID=row['algorithmID'])
                rec_create_dt = create_dt.strptime(rec.create_datetime, DATE_FMT)
                # Alter create date if earlier
                if create_dt < rec_create_dt:
                    rec.create_datetime = create_dt.strftime(DATE_FMT)
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

    # Open up NVCL data pickle files, extract create dates and create a new sqlite db
    for subdir in ['data','other','emptyrecs','nodata']:
        pkl_subdir = Path(pickle_dir) / Path(subdir)
        print("Loading", pkl_subdir)
        load_data(pkl_subdir, subdir)
    print("Done.")
