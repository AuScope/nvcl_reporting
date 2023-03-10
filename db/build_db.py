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

from peewee import SqliteDatabase, Model, TextField, DateField, CompositeKey, IntegrityError
from datetime import date


db = SqliteDatabase('nvcl2.db')

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
        primary_key = CompositeKey('state', 'nvcl_id', 'log_id', 'algorithm', 'log_type', 'algorithmID')
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
            print(f"{create_dt}")
        # Look over the rows in the pandas DataFrame
        for idx, row in df.iterrows():
            print("Inserting", row['state'], row['nvcl_id'], row['log_id'], row['algorithm'])
            try:
                # NB: mincnts is 'metres' in these old pkl files
                Meas.create(report_category=report_category, state=repr(row['state']), nvcl_id=repr(row['nvcl_id']),
                      create_datetime=repr(create_dt), log_id=repr(row['log_id']), algorithm=repr(row['algorithm']),
                      log_type=repr(row['log_type']), algorithmID=repr(row['algorithmID']),
                      minerals=repr(row['minerals']), mincnts=repr(list(row['metres'])), data=repr(row['data']))
            except IntegrityError:
                # This is a duplicate, modify create date if earlier than the stored one
                rec = Meas.get(report_category=report_category, state=repr(row['state']), nvcl_id=repr(row['nvcl_id']),
                       log_id=repr(row['log_id']), algorithm=repr(row['algorithm']), log_type=repr(row['log_type']),
                       algorithmID=repr(row['algorithmID']))
                print("DUPLICATE: ", rec, )
                rec_create_dt = eval(rec.create_datetime)
                print("Rec create dt:", rec_create_dt)
                # Alter create date if earlier
                if create_dt < rec_create_dt:
                    rec.create_datetime = repr(create_dt)
                    print("Saved", create_dt)
                    rec.save()


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
    for subdir in ['data','other','empty','nodata']:
        pkl_subdir = Path(pickle_dir) / Path(subdir)
        load_data(pkl_subdir, subdir)
    print("Done.")
