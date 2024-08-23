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

# External imports
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

import seaborn as sns
sns.set_context("talk")

# Financial year imports
import fiscalyear
fiscalyear.setup_fiscal_calendar(start_month=7)

# nvcl_kit imports
from nvcl_kit.reader import NVCLReader
from nvcl_kit.param_builder import param_builder
from nvcl_kit.constants import has_VNIR, has_SWIR, has_TIR
from multiprocessing import Pool

# Local imports
from db.readwrite_db import import_db, export_db, DF_COLUMNS
from db.tsg_metadata import TSGMeta
from calculations import calc_stats, plot_results
from constants import HEIGHT_RESOLUTION, ANALYSIS_CLASS, DATA_CATS, CONFIG_FILE, PROV_LIST, TEST_RUN
from constants import REPORT_DATE, DATA_CATS_NUMS
from helpers import conv_mindata, make_row

# Dataset dictionary - stores current NVCL datasets
g_dfs = {}

# If true, then will ignore previous downloads
SW_ignore_importedIDs = True


def update_data(prov_list: [], db_file: str, tsg_meta_file: str):
    """ Read database for any past data and poll NVCL services to see if there is any new data
        Save updates to database
        Upon keyboard interrupt save updates to database and exit

        :param prov_list: list of NVCL service providers
        :param db_file: database filename
        :param tsg_meta_file: TSG metadata filename
    """
    tsg_meta = TSGMeta(tsg_meta_file)

    MAX_BOREHOLES = 9999
    if TEST_RUN:
        # Optional maximum number of boreholes to fetch, default is no limit
        #MAX_BOREHOLES = 10
        new_prov_list = ['TAS','WA','NSW','QLD','NT','SA']
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
            param_list = [(prov, known_id_df, tsg_meta, MAX_BOREHOLES) for prov in prov_list]
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
        #    export_db(db_file, g_dfs[data_cat], data_cat, tsg_meta)
        # SIGINT is Ctrl-C
        sys.exit(int(signal.SIGINT))

    # Once finished, save out data to database
    for data_cat in DATA_CATS:
        print(f"\nSaving '{data_cat}' to {db_file}")
        export_db(db_file, g_dfs[data_cat], data_cat, tsg_meta)

def do_prov(prov, known_id_df, tsg_meta, MAX_BOREHOLES):
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
    param = param_builder(prov, max_boreholes=MAX_BOREHOLES)
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

        # Download previously unknown NVCL id dataset from service
        logs_data_list = reader.get_logs_data(nvcl_id)
        now_date = datetime.datetime.now().date()
        ###
        # If no NVCL data in this borehole, make a 'nodata' record
        ###
        if not logs_data_list:
            print(f"No NVCL data for {nvcl_id}! Inserting as 'no_data'.")
            new_row = make_row(prov, boreholes_list[idx], datetime.date.min, datetime.date.min)
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
                # If there is no 'modified_date' then use min date - i.e. year 1
                modified_date = datetime.date.min

            # Get Hylogger scan date from CSV file
            hl_scan_date = tsg_meta.get_hl_scan_date(nvcl_id)
            if hl_scan_date is None:
                # If there is no scan date, then use 'created_date' as the scan date
                created_datetime = getattr(ld, 'created_date', None)
                if isinstance(created_datetime, datetime.datetime):
                    hl_scan_date = created_datetime.date()
                else:
                    # If there is no 'created_date' or scan date, then use min date - i.e. year 1
                    hl_scan_date = datetime.date.min

            # Make a new row for insertion with hl_scan_date and modified_date
            new_row = make_row(prov, boreholes_list[idx], hl_scan_date, modified_date)

            # log types : 0=domain 1=class 2=decimal 3=image 4=profilometer 5=spectral 6=mask
            # From: https://github.com/AuScope/NVCLDataServices/blob/master/sql/Oracle/createNVCLDB11g.sql
            key = 'empty'
            if ld.log_type in DATA_CATS_NUMS:
                key = f"log{ld.log_type}"
                new_row.log_type = ld.log_type
                new_row.log_id = ld.log_id
                new_row.algorithm = ld.log_name # ???
                new_row.has_vnir = has_VNIR(new_row.algorithm)
                new_row.has_swir = has_SWIR(new_row.algorithm)
                new_row.has_tir = has_TIR(new_row.algorithm)
                new_row.algorithm_id = ld.algorithm_id
                # TODO: Expand to other types
                # If type 1 (others?) then get the mineral class data
                bh_data = reader.get_borehole_data(ld.log_id, HEIGHT_RESOLUTION, ANALYSIS_CLASS)
                if bh_data:
                    minerals, mincnts = np.unique([getattr(v, 'classText', 'Unknown') for v in bh_data.values()], return_counts=True)
                    new_row.minerals = minerals.tolist()
                    new_row.mincnts = mincnts.tolist()
                    new_row.data = conv_mindata(bh_data)
                else:
                    key = 'empty'

            # Convert to list for insertion into data frame
            new_data = new_row.as_list()

            # Add new data to the results dataframe
            results[key] = pd.concat([results[key], pd.Series(new_data, index=results[key].columns).to_frame().T], ignore_index=True)
    return results


def load_data(db_file):
    """ Load NVCL data from database file

    :param db_file: directory path of database file
    """
    print(f"Loading database {db_file}")
    for idx, data_cat in enumerate(DATA_CATS):
        g_dfs[data_cat] = import_db(db_file, data_cat)
        print(f"{idx+1} of {len(DATA_CATS)}: {data_cat} done")
    print("Loading database done.")


def load_and_check_config(config_file: str) -> dict:
    """ Loads config file
    This file contains the directories where the database file is kept, 
    TSG metadata file and the directory where the plot files are kept.

    :param config_file: config filename
    :returns: dict of config values
    """
    # Open config file
    try:
        with open(config_file, "r") as fd:
            # Parse config file
            try:
                config = yaml.safe_load(fd)
            except yaml.YAMLError as ye:
                print(f"Error in configuration file: {ye}")
                sys.exit(1)
    except OSError as oe:
        print(f"Cannot load config file {config_file}: {oe}")
        sys.exit(1)

    # If nothing in file
    if config is None:
        print(f"Cannot load config file {config_file}, it is empty")
        sys.exit(1)

    # Check keys
    for key in ('db', 'plot_dir', 'tsg_meta_file'):
        if key not in config:
            print(f"Config file {config_file} is missing a value for '{key}'")
            sys.exit(1)
        # Try to create plot directory
        if key in ('plot_dir') and not os.path.exists(config[key]):
            try:
                os.mkdir(config[key])
            except OSError as oe:
                print(f"Cannot load create directory {config[key]}: {oe}")
                sys.exit(1)
    return config


def main(sys_argv):
    """ MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN

    :param sys_argv: sys.argv from command line, can be overridden for testing purposes
    """
    # Configure command line arguments
    parser = argparse.ArgumentParser(description="NVCL Report Creator")
    parser.add_argument('-u', '--update', action='store_true', help="Update database from NVCL services")
    parser.add_argument('-f', '--full', action='store_true', help="Create full report")
    parser.add_argument('-b', '--brief', action='store_true', help="Create brief report")
    parser.add_argument('-r', '--report_date', action='store', help="Create report based on this date, format: YYYY-MM-DD")
    parser.add_argument('-d', '--db', action='store', help="Database filename")
    parser.add_argument('-c', '--config', action='store', help="Config file")

    # Parse command line arguments
    args = parser.parse_args(sys_argv[1:])

    # Complain & exit if nothing selected
    if not (args.update or args.full or args.brief):
        print("No procedural command line options were selected. Please use '--update' and/or either '--full' or '--brief'")
        parser.print_usage()
        sys.exit(1)

    # Complain if both full and brief reports were selected
    if args.full and args.brief:
        print("Cannot select both full and brief report. Please select one or the other.")
        parser.print_usage()
        sys.exit(1)

    # Complain if config file does not exist
    if args.config is not None and not os.path.isfile(args.config):
        print(f"Cannot find config file {args.config}")
        parser.print_usage()
        sys.exit(1)
    if args.config is None:
        config_file = CONFIG_FILE
    else:
        config_file = args.config 

    # Load configuration
    config = load_and_check_config(config_file)
    plot_dir = config['plot_dir']
    tsg_meta_file = config['tsg_meta_file']
    now = datetime.datetime.now()
    print("Running on", now.strftime("%A %d %B %Y %H:%M:%S"))
    sys.stdout.flush()

    data_loaded = False

    # Set report date if not supplied on command line
    report_date = REPORT_DATE
    if args.report_date:
        try:
            report_date = datetime.datetime.strptime(args.report_date, '%Y-%m-%d').date()
        except ValueError as ve:
            print(f"Report date has incorrect format: {ve}")
            sys.exit(1)

    # Assigns a database, defaults to database defined in config
    if args.db is not None:
        db = args.db
    elif 'db' in config:
        db = config['db']
    else:
        print("Database not defined in config file, nor on command line")
        sys.exit(1)
    if not os.path.exists(db):
        print(f"{db} does not exist. Will attempt to create a new one...")

    # Open database, talk to services, update database
    if args.update:
        update_data(PROV_LIST, db, tsg_meta_file)
        data_loaded = True

    # Load database from designated database
    if not data_loaded:
        load_data(db)

    # Create report
    if args.full or args.brief:
        # Create plot dir if doesn't exist
        plot_path = Path(plot_dir)
        if not plot_path.exists():
            os.mkdir(plot_dir)
        # Calculate stats for graphs
        if args.full:
            calc_stats(g_dfs, PROV_LIST, db)
        # FIXME: This is a sorting prefix, used to be pickle_dir name
        prefix = "version"
        # Create plots and report
        plot_results(report_date, g_dfs, plot_dir, prefix, args.brief)

    print("Done.")

if __name__ == "__main__":
    main(sys.argv)

