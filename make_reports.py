#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import shutil
from pathlib import Path
import argparse
import re
import yaml
import datetime
import time
import signal
from collections import OrderedDict
from itertools import zip_longest

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from wordcloud import WordCloud, STOPWORDS
from periodictable import elements

import xmltodict
import requests
import seaborn as sns
sns.set_context("talk")
import matplotlib.pyplot as plt

# Financial year imports
import fiscalyear
fiscalyear.setup_fiscal_calendar(start_month=7)
from fiscalyear import FiscalDate, FiscalQuarter

# nvcl_kit imports
from nvcl_kit.reader import NVCLReader
from nvcl_kit.param_builder import param_builder
from types import SimpleNamespace

# Local imports
from make_pdf import write_report
from db.readwrite_db import import_db, export_db, DF_COLUMNS
from make_plots import (plot_borehole_percent, plot_borehole_number, plot_borehole_kilometres, plot_wordclouds,
                  plot_geophysics, plot_elements, plot_spectrum_group, plot_algorithms)

# NVCL provider list. Format is (WFS service URL, NVCL service URL, bounding box coords)
PROV_LIST = ['NSW', 'NT', 'TAS', 'VIC', 'QLD', 'SA', 'WA']

# Configuration file
CONFIG_FILE = "config.yaml"

# Test run
TEST_RUN = True

# Maximum number of boreholes to retrieve from each provider
MAX_BOREHOLES = 999999

# Abort information file - contains the NVCL log id at which run was aborted
ABORT_FILE = Path('./run_NVCL_abort.txt')

# Report data categories
DATA_CATS = ['log1', 'log2', 'log6', 'empty', 'nodata']

# Borehole parameters
HEIGHT_RESOLUTION = 1.0
ANALYSIS_CLASS = ''

# Dataset dictionary - stores current NVCL datasets
g_dfs = {}

# Matplotlib legend positioning constant
BBX2A = (1.0, 0.5)


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


def create_stats(cdf: pd.DataFrame) -> pd.DataFrame:
    """
    Create statistics

    :param cdf: input data in a pandas dataframe
    :returns: statistics as another pandas dataframe
    """
    ca_stats = pd.DataFrame()
    nbores = cdf.loc['nbores']
    if isinstance(nbores, pd.DataFrame):
        nbores = nbores.sum()
    nbores.name = 'nbores'
    ca_stats = ca_stats.append(nbores)
    nmetres = cdf.loc['nmetres']
    if isinstance(nmetres, pd.DataFrame):
        nmetres = nmetres.sum()
    nmetres.name = 'nmetres'
    ca_stats = ca_stats.append(nmetres)
    IDs = np.unique(list(filter(lambda x: re.match(r"algorithm_\d+", x), cdf.index.tolist())))
    for ID in IDs:
        cID = cdf.loc[ID]
        if isinstance(cID, pd.DataFrame):
            cID = cID.sum()
        cID.name = ID
        ca_stats = ca_stats.append(cID)
    return(pd.DataFrame(ca_stats).transpose())


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
    MAX_BOREHOLES = 999999
    if TEST_RUN:
        # Optional maximum number of boreholes to fetch, default is no limit
        MAX_BOREHOLES = 3
        new_prov_list = ['TAS']
        prov_list = new_prov_list

    SW_ignore_importedIDs = False

    #report_category = TextField() # Can be any one of 'log1', 'log2', 'empty' and 'nodata'
    #provider = TextField()
    #nvcl_id = TextField()
    #modified_datetime = DateField()
    #log_id = TextField()
    #algorithm = TextField()
    #log_type = TextField()
    #algorithm_id = TextField()
    #minerals = TextField() # Unique minerals
    #mincnts = TextField()  # Counts of unique minerals as an array
    #data = TextField()     # Raw data as a dict

    # Compile a list of known NVCL ids from database
    known_id_list = []
    # Loop over data categories
    for data_cat in DATA_CATS:
        # Import data frame from database file
        print(f"Importing db {db_file}, {data_cat}")
        g_dfs[data_cat] = import_db(db_file, data_cat)
        #print(f"g_dfs[{data_cat}] = {g_dfs[data_cat]}")
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

            nvcl_id_list = reader.get_nvcl_id_list()
            print(f"{len(nvcl_id_list)} NVCL boreholes found for {prov}")

            # Check for no NVCL ids & skip to next service
            if not nvcl_id_list:
                print(f"!!!! Could not download NVCL ids for {prov}")
                continue

            for iID, nvcl_id in enumerate(nvcl_id_list):
                print('-'*50)
                print(f"{nvcl_id} - {prov} ({iID+1} of {len(nvcl_id_list)})")
                print('-'*10)
                current_id = nvcl_id
                # Is this a known NVCL id? Then ignore
                if (SW_ignore_importedIDs and nvcl_id in known_ids):
                    print(f"{nvcl_id} is already imported, next...")
                    continue

                # Download previously unknown NVCL id dataset from service
                logs_data_list = reader.get_logs_data(nvcl_id)
                now_date = datetime.datetime.now().date()
                # If no NVCL data, make a 'nodata' record
                if not logs_data_list:
                    print(f"No NVCL data for {nvcl_id}!") 
                    #'provider', 'nvcl_id', 'modified_datetime', 'log_id', 'algorithm', 'log_type', 'algorithm_id', 'minerals', 'mincnts', 'data'
                    data = [prov, nvcl_id, now_date, '', '', '', '', [], [], []]
                    g_dfs['nodata'] = pd.concat([g_dfs['nodata'], pd.Series(data, index=g_dfs['nodata'].columns).to_frame().T], ignore_index=True)

                # If this log has NVCL data
                for ld in logs_data_list:
                    if SW_ignore_importedIDs and \
                      ((ld.log_id in g_dfs['log1'].log_id.values) or (ld.log_id in g_dfs['empty'].log_id.values)):
                        print(f"Log id {ld.log_id} already imported, next...")
                        continue
                    minerals = []
                    #'provider', 'nvcl_id', 'modified_datetime', 'log_id', 'algorithm', 'log_type', 'algorithm_id', 'minerals', 'mincnts', 'data'
                    # If provider supports modified_date then use it
                    modified_datetime = getattr(ld, 'modified_date', now_date)
                    # print(f"From NVCL {modified_datetime=}")
                    data = [prov, nvcl_id, modified_datetime.date(), ld.log_id, ld.log_name, ld.log_type, ld.algorithm_id, [], [], []]
                    # If type 1 then get the mineral class data
                    if ld.log_type == '1':
                        bh_data = reader.get_borehole_data(ld.log_id, HEIGHT_RESOLUTION, ANALYSIS_CLASS)
                        if bh_data:
                            minerals, mincnts = np.unique([getattr(bh_data[i], 'classText', 'Unknown') for i in bh_data.keys()], return_counts=True)
                            #'provider', 'nvcl_id', 'modified_datetime', 'log_id', 'algorithm', 'log_type', 'algorithm_id', 'minerals', 'mincnts', 'data'
                            # Convert the 'minerals', 'mincnts' & 'data' fields to lists etc.
                            data = [prov, nvcl_id, modified_datetime.date(), ld.log_id, ld.log_name, ld.log_type, ld.algorithm_id, minerals.tolist(), mincnts.tolist(), conv_mindata(bh_data)]

                    # Add new data to the dataframe
                    if len(minerals) > 0:
                        key = f"log{ld.log_type}"
                        g_dfs[key] = pd.concat([g_dfs[key], pd.Series(data, index=g_dfs[key].columns).to_frame().T], ignore_index=True)
                    else:
                        g_dfs['empty'] = pd.concat([g_dfs['empty'], pd.Series(data, index=g_dfs['empty'].columns).to_frame().T], ignore_index=True)

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


def calc_bh_kms(prov: str, start_date: datetime.date = None, end_date: datetime.date = None,
                return_cnts: bool = False) -> dict:
    """
    Assemble a dict of borehole depths

    :param prov: name of data provider, state or territory
    :param start_date: optional start date, datetime.date() object
    :param end_date: optional end date, datetime.date() object
    :param return_cnts: if true will return counts
    :returns: (optional) borehole counts, a dict of borehole depths (units: kilometres, key is NVCL id)
    """
    # Filter by data provider, optional date range
    print(f"calc_bh_kms({prov})")
    if start_date is None or end_state is None:
        df = g_dfs['log1'][g_dfs['log1']['provider'] == prov]
    else:
        df = g_dfs['log1'][g_dfs['log1']['provider'] == prov and g_dfs['log1']['modified_datetime'] > start_date and g_dfs['log1']['modified_datetime'] < end_date]
    bh_kms = {}
    nvcl_ids = np.unique(df['nvcl_id'])
    cnts = len(nvcl_ids)
    for nvcl_id in nvcl_ids:
        depth_set = set()
        # Filter by NVCL id
        nvclid_df = df[df.nvcl_id == nvcl_id]
        for idx, row in nvclid_df.iterrows():
            depths = [depth for depth in row['data']]
            # Skip "Tray" algorithms 
            if row['algorithm'] != 'Tray':
                # Each element of the 'data' column is a dictionary whose key is depth
                # Create a set of depth range tuples
                depth_set.update(set(depths))

        # Divide by 1000 to convert from metres to kilometres
        depth_list = list(depth_set)
        if len(depth_list) > 0:
            bh_kms[nvcl_id] = (max(depth_list) - min(depth_list)) / 1000.0
            print(f"Borehole {nvcl_id} has {bh_kms[nvcl_id]} kms")

    if not return_cnts:
        return bh_kms
    return cnts, bh_kms


def calc_stats(prov_list: list, prefix: str):
    """
    Calculates statistics based on input dataset dictionary

        :param prov_list: list of NVCL service providers
        :param prefix: sorting prefix for reports
    """
    df_allstats = pd.DataFrame()
    # Munge data
    print(f"Calculating initial statistics ...{g_dfs}")
    # Loop around for each provider
    for prov in prov_list:
        cdf = g_dfs['log1'][g_dfs['log1']['provider'] == prov]
        if cdf.empty:
            continue

        algorithms = np.unique(cdf['algorithm'])

        # Loop around for each algorithm e.g. 'Min1 uTSAS'
        for algorithm in algorithms:
            alg_cdf = cdf[cdf.algorithm == algorithm]
            df_nbores = pd.DataFrame.from_records(zip_longest(*alg_cdf.minerals.values))
           
            # NB: This nmetres cannot be used for provider by provider totals
            df_nmetres = pd.DataFrame.from_records(zip_longest(*alg_cdf.metres.values))

            df_algorithm_id = alg_cdf.algorithm_id
            df_temp = df_nbores.apply(pd.Series.value_counts)
            df_nborescount = df_temp.sum(axis=1)

            df_cstats = pd.DataFrame(columns=['nbores', 'nmetres'])
            df_cstats['nbores'] = df_nborescount

            if len(df_algorithm_id) == 1:
                IDs = df_algorithm_id.to_list()*len(df_cstats)
            else:
                IDs = df_algorithm_id.to_list()
            algos = []
            for iC, col in df_nbores.iteritems():
                # print(IDs[iC])
                algo = f"algorithm_{IDs[iC]}"
                algos.append(algo)
                if algo not in df_cstats:
                    df_cstats.insert(0, algo, 0 * len(df_cstats))
                df_cstats[algo].loc[col.dropna().tolist()] += 1
            df_cstats[np.unique(algos)] = df_cstats[np.unique(algos)].replace({0: np.nan})

            for row, value in df_nborescount.iteritems():
                df_cstats.loc[row, 'nmetres'] = np.nansum(df_nbores.isin([row]) * df_nmetres.values)

            df_cstats = df_cstats.transpose()
            df_cstats['provider'] = prov 
            df_cstats['algorithm'] = algorithm
            df_cstats['stat'] = df_cstats.index

            df_allstats = df_allstats.append(df_cstats, ignore_index=True, sort=False)

    # Calculate algorithm statistics
    if all(stat_type in df_allstats for stat_type in ['provider', 'algorithm', 'stat']):
        df_allstats = df_allstats.set_index(['provider', 'algorithm', 'stat'])

    algorithm_stats_all = {}
    algorithms = np.unique(g_dfs['log1']['algorithm'])
    print("Calculating algorithm based statistics ...")
    for algorithm in algorithms:
        cdf = df_allstats.xs(algorithm, level='algorithm').dropna(axis=1, how='all').droplevel(0)
        algorithm_stats_all[algorithm] = create_stats(cdf)

    # Calculate algorithm by provider statistics
    algorithm_stats_byprov = {}
    print("Calculating algorithm by provider based statistics ...")
    for algorithm in algorithms:
        algorithm_stats_byprov[algorithm] = {}
        sdf = df_allstats.xs(algorithm, level='algorithm')
        providers = np.unique(g_dfs['log1'][g_dfs['log1']['algorithm'] == algorithm]['provider'])
        for prov in providers:
            cdf = sdf.xs(p, level='provider').dropna(axis=1, how='all')
            algorithm_stats_byprov[algorithm][prov] = create_stats(cdf)

    # Delete the abort file
    if ABORT_FILE.is_file():
        ABORT_FILE.unlink()

    g_dfs['stats_all'] = df_allstats
    g_dfs['stats_byalgorithms'] = algorithm_stats_all
    g_dfs['stats_byprov'] = algorithm_stats_byprov


def fill_in(src_labels, dest, dest_labels):
    """
    Fills in the missing data points when a provider is missing data

    :param src_labels: numpy array of all provider labels
    :param dest: numpy array of count labels for all providers with nonzero counts
    :param dest_labels: numpt array of provider labels, for all providers with nonzero counts
    """
    for idx in range(len(src_labels)):
        if len(dest_labels) <= idx or src_labels[idx] != dest_labels[idx]:
            dest_labels = np.insert(dest_labels, idx, src_labels[idx])
            dest = np.insert(dest, idx, 0.0)
    return dest, dest_labels

def get_fy_date_ranges() -> (datetime.date, datetime.date, datetime.date, datetime.date):
    """
    Returns four datetime.date() objects for start & end of financial year and quarter

    :returns: y_start, y_end, q_start, q_end datetime.date() objects
    """
    y = FiscalDate.today()
    py = y.prev_fiscal_year
    y_start = py.start.date()
    y_end = py.end.date()
    pq = y.prev_fiscal_quarter
    q_start = pq.start.date()
    q_end = pq.end.date()
    return y_start, y_end, q_start, q_end


def get_cnts(all_provs: list, start_date: datetime.date, end_date: datetime.date) -> (int, int):
    """
    Gets kilometres and borehole counts for a provider

    :param prov: provider string
    :param start_date: start date, datetime.date() object
    :param end_date: start date, datetime.date() object
    :returns: tuple (borehole counts, borehole kilometres)
    """
    cnts = {}
    kms = {}
    for prov in all_provs:
        cnts[prov], kms[prov] = calc_bh_kms(prov, start_date, end_date, True)
    print(cnts, kms)
    return cnts, kms


def plot_results(prefix: str, brief: bool):
    """
    Generates a set of plot files

    :param prefix: sorting prefix
    :param brief: if True will create a smaller report
    """
    # Remove old plots
    shutil.rmtree(PLOT_DIR)
    os.mkdir(PLOT_DIR)

    table_data = []
    title_list = []
    if not any(key in g_dfs and ((type(g_dfs[key]) is dict and g_dfs[key] != {}) or (isinstance(g_dfs[key], pd.DataFrame) and not g_dfs[key].empty)) for key in ('log1','log2','empty','nodata')):
        print("Datasets are empty, please create them before enabling plots")
        print(f"g_dfs.keys()={g_dfs.keys()}")
        print(f"g_dfs={g_dfs}")
        sys.exit(1)
    df_all = pd.concat([g_dfs['log1'], g_dfs['log2'], g_dfs['empty'], g_dfs['nodata']])
    dfs_log2_all = pd.concat([g_dfs['log2'], g_dfs['empty'][g_dfs['empty']['log_type'] == '2']])
    all_provs, all_counts = np.unique(df_all.drop_duplicates(subset='nvcl_id')['provider'], return_counts=True)

    if not brief:
        # Count log1 data for all providers
        log1_provs, log1_counts = np.unique(g_dfs['log1'].drop_duplicates(subset='nvcl_id')['provider'], return_counts=True)
        # Insert zeros for any providers that are missing
        if len(all_provs) > len(log1_provs):
            log1_counts, log1_provs = fill_in(all_provs, log1_counts, log1_provs)
        # Make log1 table
        make_table(table_data, title_list, list(log1_provs), list(log1_counts), "Log 1 Counts by Provider")

        # Count log2 data for all providers
        log2_provs, log2_counts = np.unique(g_dfs['log2'].drop_duplicates(subset='nvcl_id')['provider'], return_counts=True)
        # Insert zeros for any providers that are missing
        if len(all_provs) > len(log2_provs):
            log2_counts, log2_provs = fill_in(all_provs, log2_counts, log2_provs)
        # Make log2 table
        make_table(table_data, title_list, list(log2_provs), list(log2_counts), "Log 2 Counts by Provider")

        # Count 'nodata' data for all providers
        nodata_provs, nodata_counts = np.unique(g_dfs['nodata'].drop_duplicates(subset='nvcl_id')['provider'], return_counts=True)
        # Insert zeros for any provider that are missing
        if len(all_provs) > len(nodata_provs):
            nodata_counts, nodata_provs = fill_in(all_provs, nodata_counts, nodata_provs)

        # Make nodata table
        make_table(table_data, title_list, list(nodata_provs), list(nodata_counts), "'No data' Counts by Provider")

        # Count 'empty' data for all provider 
        df_empty_log1 = g_dfs['empty'][g_dfs['empty']['log_type'] == '1']
        empty_provs, empty_counts = np.unique(df_empty_log1.drop_duplicates(subset='nvcl_id')['provider'], return_counts=True)
        # Insert zeros for any providers that are missing
        if len(all_provs) > len(empty_provs):
            empty_counts, empty_provs = fill_in(all_provs, empty_counts, empty_provs)

        # Make empty counts table
        make_table(table_data, title_list, list(empty_provs), list(empty_counts), "'empty' Counts by Provider")

        # Plot percentage of boreholes by provider and data present
        plot_borehole_percent(nodata_counts, log1_counts, all_counts, log1_provs, nodata_provs, empty_provs)

    # Plot number of boreholes by provider
    plot_borehole_number(all_provs, all_counts)
    
    # Table of number of boreholes by provider
    make_table(table_data, title_list, list(all_provs), list(all_counts), "Number of boreholes by Provider")

    # Calculate a list of number of kilometres, one value for each provider 
    nkilometres_totals = [ sum(calc_bh_kms(prov).values()) for prov in all_provs ]

    # Make number of kilometres by provider table
    make_table(table_data, title_list, list(all_provs), nkilometres_totals, "Number of borehole kilometres by Provider")
    
    # Plot borehole kilometres by provider
    plot_borehole_kilometres(all_provs, nkilometres_totals)

    # Calculate borehole counts and kilometres
    y_start, y_end, q_start, q_end = get_fy_date_ranges()
    y_cnts, y_kilometres = get_cnts(all_provs, y_start, y_end)
    q_cnts, q_kilometres = get_cnts(all_provs, q_start, q_end)

    # Pretty printed date strings
    y_date = y_end.strftime('(%d/%m/%Y)')
    q_date = q_end.strftime('(%d/%m/%Y)')
    y_date_pretty = y_end.strftime("%A %d %b %Y")
    q_date_pretty = q_end.strftime("%A %d %b %Y")

    # All possible providers are taken from annual data
    all_keys = list(y_cnts.keys())

    # Plot yearly and quarterly comparisons for counts by provider
    all_cnts_dict = dict(zip(all_provs, all_counts))
    q_diffs = calc_metric_diffs(all_keys, all_cnts_dict, q_cnts)
    y_diffs = calc_metric_diffs(all_keys, all_cnts_dict, y_cnts)
    plot_borehole_number(all_keys, y_diffs, title=f"Borehole counts since end of last financial year {y_date}", filename="borehole_number_y.png")
    plot_borehole_number(all_keys, q_diffs, title=f"Borehole counts since last quarter {q_date}", filename="borehole_number_q.png")

    # Tabulate yearly and quarterly comparisons for counts by provider
    make_table(table_data, title_list, list(all_keys), q_diffs, f"Number of boreholes by Provider since last quarter {q_date}")
    make_table(table_data, title_list, list(all_keys), y_diffs, f"Number of boreholes by Provider since end of last financial year {y_date}")

    # Plot yearly and quarterly comparisons for kilometres by provider
    nkilometres_dict = dict(zip(all_provs, nkilometres_totals))
    print("nkilometres_dict = ", nkilometres_dict)
    print("q_kilometres = ", q_kilometres)
    print("y_kilometres = ", y_kilometres)
    q_diffs = calc_metric_diffs(all_keys, nkilometres_dict, q_kilometres)
    y_diffs = calc_metric_diffs(all_keys, nkilometres_dict, y_kilometres)
    plot_borehole_kilometres(all_keys, y_diffs, title=f"Borehole kilometres since end of last financial year {y_date}", filename="borehole_kilometres_y.png")
    plot_borehole_kilometres(all_keys, q_diffs, title=f"Borehole kilometres since last quarter {q_date}", filename="borehole_kilometres_q.png")
 
    # Tabulate yearly and quarterly comparisons for kilometres by provider
    make_table(table_data, title_list, list(all_keys), q_diffs, f"Number of borehole kilometres by Provider since last quarter {q_date}")
    make_table(table_data, title_list, list(all_keys), y_diffs, f"Number of borehole kilometres by Provider since end of last financial year {y_date}")

    # Plot word clouds
    # plot_wordclouds(dfs_log2_all)

    if not brief:

        # Get algorithms from any available service
        algo2 = {}
        for prov in PROV_LIST:
            param = param_builder(prov, max_boreholes=MAX_BOREHOLES)
            if not param:
                print(f"Cannot build parameters for {prov}: {param}")
                continue

            print(f"param={param}")
            reader = NVCLReader(param)
            if not reader.wfs:
                print(f"ERROR! Cannot connect to {prov}")
                continue
            algo2 = reader.get_algorithms()
            if len(algo2.keys()) > 0:
                break

        algo2['0'] = '0'

        # Plot algorithms
        plot_algorithms(algo2)

        # Plot uTSAS graphs
        plot_spectrum_group(algo2, prefix)

        '''
        pd.DataFrame({'algo':grp_algos, 'counts':grp_counts}).plot.bar(x='algo',y='counts', rot=20)
        [min_algos, min_counts] = np.unique(g_dfs['log1'][g_dfs['log1'].algorithm.str.startswith('Min')]['algorithm'], return_counts=True)
        pd.DataFrame({'algo':min_algos, 'counts':min_counts}).plot.bar(x='algo',y='counts', rot=20)
        index = [x.replace('Grp','') for x in grp_algos]
        df = pd.DataFrame({'Grp': grp_counts, 'Min': min_counts}, index=index).plot.bar(rot=20)
        '''
        # Plot element graphs
        plot_elements(dfs_log2_all)

        # Plot geophysics
        plot_geophysics(dfs_log2_all)

    now = datetime.datetime.now()
    metadata = { "Authors": "Vincent Fazio & Shane Mule",
                 "Sources": "This report was compiled from NVCL datasets downloaded from the NVCL nodes managed\nby the state geological surveys of QLD, NSW, Vic, Tas, NT, SA & WA", 
                 "Report Date": now.strftime("%A %b %d %Y"),
                 "Using quarterly data from": q_date_pretty,
                 "Using EOFY data from": y_date_pretty }

    # Finally write out pdf report
    if brief:
        write_report("report-brief.pdf", PLOT_DIR, table_data, title_list, metadata, brief)
    else:
        write_report("report.pdf", PLOT_DIR, table_data, title_list, metadata, brief)


def calc_metric_diffs(all_keys, larger, smaller):
    '''
    Calculate numerical difference between dicts of metrics, key is provider name
    NB: if value is missing zero is returned

    :param all_keys: list of all possible dict keys (provider names)
    :param larger: dict of metric values keys are provider names
    :param smaller: dict of metric values keys are provider names
    :returns: array of differences, one for each value in 'all_keys'
    '''
    result = []
    for key in all_keys:
        if key in larger and key in smaller and larger[key] > smaller[key]:
            result.append(larger[key] - smaller[key])
        else:
            result.append(0)
    return result    


def make_table(table_data, title_list, table_header, table_datarows, title):
    """ Makes a table data structure for passing to report building routines

    :param table_data: list passed to report building routines
    :param title_list: list of table titles
    :param table_header: list of table header strings
    :param table_datarow: list of data rows to be inserted into table, each row same length as headers string list
    """
    table_rows = [table_header]
    table_rows.append(table_datarows)
    table_data.append(table_rows)
    title_list.append(title)


def dfcol_algoid2ver(df, algoid2ver):
    """ Renames columns in dataframe from algorithm id to version id

    :param df: pandas dataframe
    :param algoid2ver: dictionary of algorithm id -> version id
    :returns: transformed dataframe
    """
    df = df.rename({y: re.sub(r'algorithm_(\d+)', lambda x: f"version_{algoid2ver.get(x.group(1), '0')}", y) for y in df.columns}, axis='columns')
    df = df.fillna(0).groupby(level=0).sum()
    return df


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
    PLOT_DIR = config['plot_dir']

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
        print(f"{db} does not exist")
        sys.exit(1)

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
        calc_stats(PROV_LIST, db)
        stats_loaded = True

    # Plot results
    if args.plot or args.brief_plot:
        # Create plot dir if doesn't exist
        plot_path = Path(PLOT_DIR)
        if not plot_path.exists():
            os.mkdir(PLOT_DIR)
        if not stats_loaded:
            calc_stats(PROV_LIST, db)
        # FIXME: This is a sorting prefix, used to be pickle_dir name
        prefix = "unknown"
        plot_results(prefix, args.brief_plot)

    print("Done.")
