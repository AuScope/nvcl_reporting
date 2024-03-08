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
from multiprocessing import Pool

# Local imports
from db.readwrite_db import import_db, export_db, DF_COLUMNS
from db.schema import DF_Row
from db.tsg_metadata import TSGMeta
from reports import calc_stats, plot_results
from constants import HEIGHT_RESOLUTION, ANALYSIS_CLASS, ABORT_FILE, DATA_CATS, CONFIG_FILE, PROV_LIST, TEST_RUN
from constants import MAX_BOREHOLES

# Dataset dictionary - stores current NVCL datasets
g_dfs = {}

# If true, then will ignore previous downloads
SW_ignore_importedIDs = True


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
    """ Convert from EPSG:4326 WGS84 (units: degrees) to EPSG:7842 GDA2020 (units: metres)
    """
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

    if TEST_RUN:
        # Optional maximum number of boreholes to fetch, default is no limit
        MAX_BOREHOLES = 2
        new_prov_list = ['TAS', 'WA','NSW','QLD','VIC', 'NT']
        prov_list = new_prov_list


    # Compile a dataframe of known NVCL ids & providers to avoid duplicates
    known_id_df = pd.DataFrame()
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
        known_id_df = pd.concat([known_id_df, g_dfs[data_cat].filter(items=['provider', 'nvcl_id']).drop_duplicates()]).reset_index(drop=True)
    else:
        # Doesn't exist? Create a new data frame
        g_dfs[data_cat] = pd.DataFrame(columns=DF_COLUMNS)

    # TODO: Reinstate later
    #if ABORT_FILE.is_file():
    #    with open(ABORT_FILE, 'r') as f:
    #        remove = f.readlines()
    #        known_ids = np.delete(known_ids, np.argwhere(known_ids == remove))
    #    # Delete the abort file
    #    ABORT_FILE.unlink()

    print("Reading NVCL data services ...")
    # Read data from NVCL services
    try:
        # Run each provider in parallel
        with Pool(processes=4) as pool:
            param_list = [(prov, known_id_df, tsg_meta) for prov in prov_list]
            prov_df_list = pool.starmap(do_prov, param_list)
            # Append new results from a provider to the global dataframe
            for prov_df in prov_df_list:
                for data_cat in DATA_CATS:
                    g_dfs[data_cat] = pd.concat([g_dfs[data_cat], prov_df[data_cat]], ignore_index=True)

    ## If user presses Ctrl-C then save out data to db & exit
    except KeyboardInterrupt:
        # TODO: Reinstate later
        ## Save current NVCL id to abort file, so we can exclude it later on
        #if current_id != '':
        #    with open(ABORT_FILE, 'w') as f:
        #        f.write(current_id)
        ## Save out data & exit
        #for data_cat in DATA_CATS:
        #    export_db(db_file, g_dfs[data_cat], data_cat, known_id_df)
        # SIGINT is Ctrl-C
        sys.exit(int(signal.SIGINT))

    # Once finished, save out data to database
    for data_cat in DATA_CATS:
        print(f"\nSaving '{data_cat}' to {db_file}")
        export_db(db_file, g_dfs[data_cat], data_cat, known_id_df)


def do_prov(prov, known_id_df, tsg_meta):
    """ Ask a provider for NVCL data, runs in its own process

    :param prov: name of provider, e.g. 'NSW'
    :returns: pandas DataFrame with columns defined by DF_COLUMNS
    """
    print('\n'+'>'*15+f"    {prov}    "+'<'*15)

    # Create results - a dict of empty dataframes
    results = {}
    for data_cat in DATA_CATS:
        results[data_cat] = pd.DataFrame(columns=DF_COLUMNS)

    # Create parameters for NVCL services
    param = param_builder(prov, max_boreholes=2)
    if not param:
        print(f"Cannot build parameters for {prov}: {param}")
        return results

    # Instantiate class and search for boreholes
    #print(f"param={param}")
    reader = NVCLReader(param)
    if not reader.wfs:
        print(f"ERROR! Cannot connect to {prov}")
        return results

    # Search for NVCL boreholes
    boreholes_list = reader.get_boreholes_list()
    nvcl_id_list = [ bh['nvcl_id'] for bh in boreholes_list ]
    print(f"{len(nvcl_id_list)} NVCL boreholes found for {prov}")

    # Check for no NVCL ids & skip to next service
    if not nvcl_id_list:
        print(f"!!!! Could not download NVCL ids for {prov}")
        return results

    for idx, nvcl_id in enumerate(nvcl_id_list):
        print('-'*50)
        print(f"{nvcl_id} - {prov} ({idx+1} of {len(nvcl_id_list)})")
        print('-'*10)
        # Is this a known NVCL id? Then ignore
        if (SW_ignore_importedIDs and len(known_id_df.query(f"nvcl_id == '{nvcl_id}' and provider == '{prov}'")) > 0):
            print(f"{nvcl_id} in {prov} is already imported, next...")
            continue

        easting, northing = to_metres(boreholes_list[idx]['x'], boreholes_list[idx]['y'])

        # Download previously unknown NVCL id dataset from service
        logs_data_list = reader.get_logs_data(nvcl_id)
        now_datetime = datetime.datetime.now()
        ###
        # If no NVCL data in this borehole, make a 'nodata' record
        ###
        if not logs_data_list:
            print(f"No NVCL data for {nvcl_id}! Inserting as 'no_data'.")
            new_row = DF_Row(provider=prov,
               borehole_id=nvcl_id,
               drill_hole_name=boreholes_list[idx]['name'],
               hl_scan_date=now_datetime.date(),
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
               modified_datetime=now_datetime.date(),
               log_id='',
               algorithm='',
               log_type='',
               algorithm_id='',
               minerals=[],
               mincnts=[],
               data=[])
            #print("AS_LIST:", new_row.as_list())
            results['nodata'] = pd.concat([results['nodata'], pd.Series(new_row.as_list(), index=results['nodata'].columns).to_frame().T], ignore_index=True)
            continue

        ###
        # If this borehole has NVCL data
        ###
        for ld in logs_data_list:
            if SW_ignore_importedIDs and \
              ((ld.log_id in g_dfs['log1'].log_id.values) or (ld.log_id in g_dfs['empty'].log_id.values)):
                print(f"Log id {ld.log_id} already imported, next...")
                continue
            minerals = []
            # If provider supports modified_date then use it
            modified_datetime = getattr(ld, 'modified_date', None)
            if isinstance(modified_datetime, datetime.datetime):
                modified_date = modified_datetime.date()
            else:
                modified_date = now_datetime.date()

            # print(f"From NVCL {modified_date=}")

            # Get Hylogger scan date from CSV file
            if nvcl_id in tsg_meta.dt_lkup:
                hl_scan_date = datetime.datetime.strptime(tsg_meta.dt_lkup[nvcl_id], '%Y-%m-%d %H:%M:%S').date()
            else:
                hl_scan_date = modified_date
            # print(f"HYLOGGER SCAN DATE {hl_scan_date=}")
            assert isinstance(hl_scan_date, datetime.date)

            # When there is no modified datetime, but there is Hylogger scan date, use the scan date
            if modified_date == now_datetime.date() and hl_scan_date < now_datetime.date():
                modified_date = hl_scan_date

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
               modified_datetime=modified_date,
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

            # Add new data to the results dataframe
            if len(minerals) > 0:
                key = f"log{ld.log_type}"
                # print(f"Adding row to dataframe at {key}")
                results[key] = pd.concat([results[key], pd.Series(new_data, index=results[key].columns).to_frame().T], ignore_index=True)
            else:
                # print("Adding row to dataframe at 'empty'")
                results['empty'] = pd.concat([results['empty'], pd.Series(new_data, index=results['empty'].columns).to_frame().T], ignore_index=True)
    return results


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
