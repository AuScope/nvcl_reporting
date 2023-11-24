#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Python imports
import sys
import os
from pathlib import Path
import argparse
import yaml
import datetime
import signal
from collections import OrderedDict

# External imports
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

import seaborn as sns
sns.set_context("talk")

from pyproj.transformer import Transformer

# Financial year imports
import fiscalyear
fiscalyear.setup_fiscal_calendar(start_month=7)

# nvcl_kit imports
from nvcl_kit.reader import NVCLReader
from nvcl_kit.param_builder import param_builder
from nvcl_kit.constants import has_VNIR, has_SWIR, has_TIR
from types import SimpleNamespace

# Local imports
from db.readwrite_db import import_db, export_db, DF_COLUMNS
from db.schema import DF_Row
from db.tsg_metadata import TSGMeta
from reports import calc_stats, plot_results
from constants import HEIGHT_RESOLUTION, ANALYSIS_CLASS, ABORT_FILE, DATA_CATS, CONFIG_FILE, PROV_LIST, TEST_RUN

# Dataset dictionary - stores current NVCL datasets
g_dfs = {}


'''
Internal functions
'''

def conv_mindata(mindata: OrderedDict) -> list:
    """ Convert mineral data to a list e.g.
    from: OrderedDict([(0.5, namespace(className='', classCount=75, classText='Andesite lava breccia', colour=(0.21568627450980393, 1.0, 0.0, 1.0))), 
    to: [[0.5, {'className': '', 'classText': 'Jarosite', 'colour': [0.40784313725490196, 0.3764705882352941, 0.10588235294117647, 1.0]}],
    """
    data_list = []
    if isinstance(mindata, OrderedDict):
        for depth, obj in mindata.items():
            # vars() converts Namespace -> dict
            if isinstance(obj, SimpleNamespace):
                data_list.append([depth, vars(obj)])
            elif isinstance(obj, list) and len(obj) == 0:
                continue
            else:
                print(repr(obj), type(obj))
                print("ERROR unknown obj type in 'data' var")
                sys.exit(1)
    else:
        print(f"ERROR {mindata} is not an ordered list")
        sys.exit(8)
    return data_list


def to_metres(x: float, y: float) -> (float, float):
    transformer = Transformer.from_crs(4326, 7842)
    return transformer.transform(y, x)


'''
Primary functions
'''

def update_data(prov_list: [], db_file: str):
    """ Read database for any past data and poll NVCL services to see if there is any new data
        Save updates to database
        Upon keyboard interrupt save updates to database and exit

        :param prov_list: list of NVCL service providers
        :param db_file: database filename
    """
    tsg_meta = TSGMeta(os.path.join("db","metadata.csv"))

    MAX_BOREHOLES = 999999
    if TEST_RUN:
        # Optional maximum number of boreholes to fetch, default is no limit
        MAX_BOREHOLES = 10
        new_prov_list = ['WA']
        prov_list = new_prov_list

    # If true, then will ignore previous downloads
    SW_ignore_importedIDs = False

    # Compile a list of known NVCL ids from database
    known_id_list = []
    # Loop over data categories
    for data_cat in DATA_CATS:
        # Import data frame from database file
        print(f"Importing db {db_file}, {data_cat}")
        g_dfs[data_cat] = import_db(db_file, data_cat)
        # print(f"g_dfs[{data_cat}] = {g_dfs[data_cat]}")
        # Check column values
        s1 = set(list(g_dfs[data_cat].columns))
        s2 = set(DF_COLUMNS)
        if s1 != s2:
            print(f"Cannot read database file {db_file}, wrong columns: {s1} != {s2}")
            sys.exit(1)
        known_id_list = np.append(known_id_list, g_dfs[data_cat].nvcl_id.values)
    else:
        # Doesn't exist? Create a new data frame
        g_dfs[data_cat] = pd.DataFrame(columns=DF_COLUMNS)

    # Remove duplicates
    known_ids = np.array(list(set(known_id_list)))

    # Remove all the NVCL ids in the abort file from known_ids list
    if ABORT_FILE.is_file():
        with open(ABORT_FILE, 'r') as f:
            remove = f.readlines()
            known_ids = np.delete(known_ids, np.argwhere(known_ids == remove))
        # Delete the abort file
        ABORT_FILE.unlink()

    print("Reading NVCL data services ...")
    # Read data from NVCL services
    current_id = ''
    try:
        # Loop over the providers
        for prov in prov_list:
            print('\n'+'>'*15+f"    {prov}    "+'<'*15)
            param = param_builder(prov, max_boreholes=MAX_BOREHOLES)
            if not param:
                print(f"Cannot build parameters for {prov}: {param}")
                continue

            # Instantiate class and search for boreholes
            print(f"param={param}")
            reader = NVCLReader(param)

            if not reader.wfs:
                print(f"ERROR! Cannot connect to {prov}")
                continue

            boreholes_list = reader.get_boreholes_list()
            nvcl_id_list = [ bh['nvcl_id'] for bh in boreholes_list ]
            print(f"{len(nvcl_id_list)} NVCL boreholes found for {prov}")

            # Check for no NVCL ids & skip to next service
            if not nvcl_id_list:
                print(f"!!!! Could not download NVCL ids for {prov}")
                continue

            for idx, nvcl_id in enumerate(nvcl_id_list):
                print('-'*50)
                print(f"{nvcl_id} - {prov} ({idx+1} of {len(nvcl_id_list)})")
                print('-'*10)
                current_id = nvcl_id
                # Is this a known NVCL id? Then ignore
                if (SW_ignore_importedIDs and nvcl_id in known_ids):
                    print(f"{nvcl_id} is already imported, next...")
                    continue

                easting, northing = to_metres(boreholes_list[idx]['x'], boreholes_list[idx]['y'])

                # Download previously unknown NVCL id dataset from service
                logs_data_list = reader.get_logs_data(nvcl_id)
                now_datetime = datetime.datetime.now()
                ###
                # If no NVCL data, make a 'nodata' record
                ###
                if not logs_data_list:
                    print(f"No NVCL data for {nvcl_id}!") 
                    new_row = DF_Row(provider=prov,
                       borehole_id=nvcl_id,
                       drill_hole_name=boreholes_list[idx]['name'],
                       hl_scan_date=now_datetime,
                       easting=easting,
                       northing=northing,
                       crs="EPSG:7842",
                       start_depth=0,
                       end_depth=boreholes_list[idx]['boreholeLength_m'],
                       has_vnir=False,
                       has_swir=False,
                       has_tir=False,
                       has_mir=False,
                       nvcl_id=nvcl_id,
                       modified_datetime=now_datetime.now(),
                       log_id='',
                       algorithm='',
                       log_type='',
                       algorithm_id='',
                       minerals=[],
                       mincnts=[],
                       data=[])
                    #print("AS_LIST:", new_row.as_list())
                    g_dfs['nodata'] = pd.concat([g_dfs['nodata'], pd.Series(new_row.as_list(), index=g_dfs['nodata'].columns).to_frame().T], ignore_index=True)

                ###
                # If this log has NVCL data
                ###
                for ld in logs_data_list:
                    if SW_ignore_importedIDs and \
                      ((ld.log_id in g_dfs['log1'].log_id.values) or (ld.log_id in g_dfs['empty'].log_id.values)):
                        print(f"Log id {ld.log_id} already imported, next...")
                        continue
                    minerals = []
                    # If provider supports modified_date then use it
                    modified_datetime = getattr(ld, 'modified_date', now_datetime)
                    print(f"From NVCL {modified_datetime=}")

                    # Get Hylogger scan date from CSV file
                    if nvcl_id in tsg_meta.dt_lkup:
                        hl_scan_date = datetime.datetime.strptime(tsg_meta.dt_lkup[nvcl_id], '%Y-%m-%d %H:%M:%S')
                    else:
                        hl_scan_date = modified_datetime
                    print(f"HYLOGGER SCAN DATE {hl_scan_date=}")

                    # When there is no modified datetime, but there is Hylogger scan date, use the scan date
                    if modified_datetime == now_datetime and hl_scan_date < now_datetime:
                        modified_datetime = hl_scan_date
                        
                    new_row = DF_Row(provider=prov,
                       borehole_id=boreholes_list[idx]['nvcl_id'],
                       drill_hole_name=boreholes_list[idx]['name'],
                       hl_scan_date=hl_scan_date,
                       easting=easting,
                       northing=northing,
                       crs="EPSG:7842",
                       start_depth=0,
                       end_depth=boreholes_list[idx]['boreholeLength_m'],
                       has_vnir=False,
                       has_swir=False,
                       has_tir=False,
                       has_mir=False,
                       nvcl_id=nvcl_id,
                       modified_datetime=modified_datetime.date(),
                       log_id='',
                       algorithm='',
                       log_type='',
                       algorithm_id='',
                       minerals=[],
                       mincnts=[],
                       data=[])
                    # If type 1 then get the mineral class data
                    if ld.log_type == '1':
                        bh_data = reader.get_borehole_data(ld.log_id, HEIGHT_RESOLUTION, ANALYSIS_CLASS)
                        if bh_data:
                            minerals, mincnts = np.unique([getattr(bh_data[i], 'classText', 'Unknown') for i in bh_data.keys()], return_counts=True)
                            new_row.log_id = ld.log_id
                            new_row.algorithm = ld.log_name # ???
                            new_row.has_vnir = has_VNIR(new_row.algorithm)
                            new_row.has_swir = has_SWIR(new_row.algorithm)
                            new_row.has_tir = has_TIR(new_row.algorithm)
                            new_row.log_type = ld.log_type
                            new_row.algorithm_id = ld.algorithm_id
                            new_row.minerals = minerals.tolist()
                            new_row.mincnts = mincnts.tolist()
                            new_row.data = conv_mindata(bh_data)
                    new_data = new_row.as_list()

                    # Add new data to the dataframe
                    if len(minerals) > 0:
                        key = f"log{ld.log_type}"
                        g_dfs[key] = pd.concat([g_dfs[key], pd.Series(new_data, index=g_dfs[key].columns).to_frame().T], ignore_index=True)
                    else:
                        g_dfs['empty'] = pd.concat([g_dfs['empty'], pd.Series(new_data, index=g_dfs['empty'].columns).to_frame().T], ignore_index=True)

                # Append new NVCL id to list of known NVCL ids
                np.append(known_ids, nvcl_id)

    # If user presses Ctrl-C then save out data to db & exit
    except KeyboardInterrupt:
        # Save current NVCL id to abort file, so we can exclude it later on
        if current_id != '':
            with open(ABORT_FILE, 'w') as f:
                f.write(current_id)
        # Save out data & exit
        for data_cat in DATA_CATS:
            export_db(db_file, g_dfs[data_cat], data_cat, known_ids)
        # SIGINT is Ctrl-C
        sys.exit(int(signal.SIGINT))

    # Once finished, save out data to database
    for data_cat in DATA_CATS:
        export_db(db_file, g_dfs[data_cat], data_cat, known_ids)


def load_data(db_file):
    """ Load NVCL data from database file

    :param db_file: directory path of database file
    """
    print(f"Loading database {db_file}")
    for data_cat in DATA_CATS:
        g_dfs[data_cat] = import_db(db_file, data_cat)


def load_and_check_config():
    """ Loads config file
    This file contains the directories where the database file is kept, 
    and the directory where the plot files are kept.
    """

    try:
        with open(CONFIG_FILE, "r") as fd:
            config = yaml.safe_load(fd)
    except OSError as oe:
        print(f"Cannot load config file {CONFIG_FILE}: {oe}")
        sys.exit(1)
    # Check keys
    for key in ('db', 'plot_dir'):
        if key not in config:
            print(f"config file {CONFIG_FILE} is missing a value for '{key}'")
            sys.exit(1)
        if key in ('plot_dir') and not os.path.exists(config[key]):
            try:
                os.mkdir(config[key])
            except OSError as oe:
                print(f"Cannot load create directory {config[key]}: {oe}")
                sys.exit(1)
    return config

if __name__ == "__main__":
    # Load configuration
    config = load_and_check_config()
    plot_dir = config['plot_dir']

    # Configure command line arguments
    parser = argparse.ArgumentParser(description="NVCL report data creator")
    parser.add_argument('-u', '--update', action='store_true', help="Update database from NVCL services")
    parser.add_argument('-s', '--stats', action='store_true', help="Calculate statistics")
    parser.add_argument('-p', '--plot', action='store_true', help="Create plots & report")
    parser.add_argument('-b', '--brief_plot', action='store_true', help="Create brief plots & report")
    parser.add_argument('-l', '--load', action='store_true', help="Load data from database")

    parser.add_argument('-d', '--db', action='store', help="Database filename.")

    # Parse command line arguments
    args = parser.parse_args()

    # Complain & exit if nothing selected
    if not (args.update or args.stats or args.plot or args.brief_plot or args.load):
        print("No instructional options were selected. What should I do? Please select an option.")
        parser.print_usage()
        sys.exit(1)

    now = datetime.datetime.now()
    print("Running on", now.strftime("%A %d %B %Y %H:%M:%S"))
    sys.stdout.flush()

    data_loaded = False
    stats_loaded = False

    # Assigns a database, defaults to database defined in config
    if args.db is not None:
        db = args.db
    elif 'db' in config:
        db = config['db']
    else:
        print("Database not defined in config file, nor on command line")
        sys.exit(1)
    if not os.path.exists(db):
        print(f"{db} does not exist. Creating new one...")

    # Open database, talk to services, update database
    if args.update:
        update_data(PROV_LIST, db)
        data_loaded = True

    # Load database from designated database
    if not data_loaded:
        load_data(db)

    # Update/calculate statistics
    if args.stats:
        # Calculate stats
        calc_stats(g_dfs, PROV_LIST, db)
        stats_loaded = True

    # Plot results
    if args.plot or args.brief_plot:
        # Create plot dir if doesn't exist
        plot_path = Path(plot_dir)
        if not plot_path.exists():
            os.mkdir(plot_dir)
        if not stats_loaded:
            calc_stats(g_dfs, PROV_LIST, db)
        # FIXME: This is a sorting prefix, used to be pickle_dir name
        prefix = "version"
        plot_results(g_dfs, plot_dir, prefix, args.brief_plot)

    print("Done.")
