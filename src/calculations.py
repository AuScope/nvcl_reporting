
# Python imports
import re
import datetime
import shutil
import os
import sys
from types import SimpleNamespace

# External imports
import pandas as pd
pd.options.mode.chained_assignment = None

import numpy as np
from itertools import zip_longest
from fiscalyear import fiscal_calendar, FiscalDate, FiscalYear, FiscalQuarter

# Local imports
from plots import Plots
from pdf import write_report

from report_table_data import ReportTableData


def calc_bh_depths(dfs: dict[str:pd.DataFrame], prov: str, date_fieldname: str,  start_date: datetime.date = None,
                end_date: datetime.date = None, return_cnts: bool = False) -> float:
    """
    Calculate depth by assembling dicts of borehole depths (and optional counts) from start date to end date
    Uses 'log1' mineral depths to calculate depths

    :param dfs: source dataframe dict
    :param prov: name of data provider, state or territory
    :param date_fieldname: name of date field to use in calculations
    :param start_date: optional start date, datetime.date() object
    :param end_date: optional end date, datetime.date() object
    :param return_cnts: if true will return counts
    :returns: (optional) borehole counts, sum of borehole depths (units: kilometres)
    """
    # Filter by data provider, optional date range
    if dfs['log1'].empty:
        if not return_cnts:
            return 0.0
        return 0, 0.0

    if start_date is None and end_date is None:
        # Has neither start_date nor end_date
        df = dfs['log1'][dfs['log1']['provider'] == prov]
    elif start_date is None:
        # Only has end_date
        df = dfs['log1'][(dfs['log1']['provider'] == prov) & (dfs['log1'][date_fieldname] < end_date)]
    elif end_date is None:
        # Only has start_date
        df = dfs['log1'][(dfs['log1']['provider'] == prov) & (dfs['log1'][date_fieldname] > start_date)]
    else:
        # Has both start_date and end_date
        df = dfs['log1'][(dfs['log1']['provider'] == prov) & (dfs['log1'][date_fieldname] > start_date) & (dfs['log1'][date_fieldname] < end_date)]
    bh_kms = {}
    nvcl_ids = np.unique(df['nvcl_id'])
    cnts = len(nvcl_ids)
    for nvcl_id in nvcl_ids:
        depth_set = set()
        # Filter by NVCL id
        nvclid_df = df[df.nvcl_id == nvcl_id]
        for idx, row in nvclid_df.iterrows():
            depths = [depth for depth, minerals in row['data']]
            # Skip "Tray" algorithms etc.
            if row['algorithm'] not in ['Tray', 'Domain', 'HoleID', 'HyLogDiag', '', 'Rockmarks']:
                # Each element of the 'data' column is a dictionary whose key is depth
                # Create a set of depth range tuples
                depth_set.update(set(depths))

        # Divide by 1000 to convert from metres to kilometres
        depth_list = list(depth_set)
        if len(depth_list) > 0:
            bh_kms[nvcl_id] = (1.0 + max(depth_list) - min(depth_list)) / 1000.0

    if not return_cnts:
        return sum(bh_kms.values())
    return cnts, sum(bh_kms.values())


def calc_stats(dfs: dict[str:pd.DataFrame], prov_list: list, prefix: str):
    """
    Calculates statistics based on input dataset dictionary

    :param dfs: source dataframe dict
    :param prov_list: list of NVCL service providers
    :param prefix: sorting prefix for reports
    """
    df_allstats = pd.DataFrame()
    # Munge data
    print(f"Calculating initial statistics ... ")
    # Loop around for each provider
    for prov in prov_list:
        # Get all rows for a provider
        cdf = dfs['log1'][dfs['log1']['provider'] == prov]
        if cdf.empty:
            continue

        # Make a list of algorithms
        algorithms = np.unique(cdf['algorithm'])

        # Loop around for each algorithm e.g. 'Min1 uTSAS'
        for algorithm in algorithms:
            alg_cdf = cdf[cdf.algorithm == algorithm]
            df_nbores = pd.DataFrame.from_records(zip_longest(*alg_cdf.minerals.values))

            df_mincnts = pd.DataFrame.from_records(zip_longest(*alg_cdf.mincnts.values))

            df_algorithm_id = alg_cdf.algorithm_id
            df_temp = df_nbores.apply(pd.Series.value_counts)
            df_nborescount = df_temp.sum(axis=1)

            df_cstats = pd.DataFrame(columns=['nbores', 'mincnts'])
            df_cstats['nbores'] = df_nborescount

            if len(df_algorithm_id) == 1:
                IDs = df_algorithm_id.to_list()*len(df_cstats)
            else:
                IDs = df_algorithm_id.to_list()
            algos = []
            for iC, col in df_nbores.items():
                algo = f"algorithm_{IDs[iC]}"
                algos.append(algo)
                if algo not in df_cstats:
                    df_cstats.insert(0, algo, 0 * len(df_cstats))
                df_cstats.loc[col.dropna().tolist(), algo] += 1
            df_cstats[np.unique(algos)] = df_cstats[np.unique(algos)].replace({0: np.nan})

            for row, value in df_nborescount.items():
                df_cstats.loc[row, 'mincnts'] = np.nansum(df_nbores.isin([row]) * df_mincnts.values)

            df_cstats = df_cstats.transpose()
            df_cstats['provider'] = prov 
            df_cstats['algorithm'] = algorithm
            df_cstats['stat'] = df_cstats.index

            df_allstats = pd.concat([df_allstats, df_cstats], ignore_index=True)

    # Calculate algorithm statistics
    if all(stat_type in df_allstats for stat_type in ['provider', 'algorithm', 'stat']):
        df_allstats = df_allstats.set_index(['provider', 'algorithm', 'stat'])

    algorithm_stats_all = {}
    algorithms = np.unique(dfs['log1']['algorithm'])
    print("Calculating algorithm based statistics ...")
    # Loop over algorithms
    for algorithm in algorithms:
        # Pull out all rows with a certain algorithm
        cdf = df_allstats.xs(algorithm, level='algorithm').dropna(axis=1, how='all').droplevel(0)
        algorithm_stats_all[algorithm] = create_stats(cdf)

    # Calculate algorithm by provider statistics
    algorithm_stats_byprov = {}
    print("Calculating algorithm by provider based statistics ...")
    # Loop over algorithms
    for algorithm in algorithms:
        algorithm_stats_byprov[algorithm] = {}
        # Pull out all rows with a certain algorithm
        sdf = df_allstats.xs(algorithm, level='algorithm')
        # Derive unique providers
        providers = np.unique(dfs['log1'][dfs['log1']['algorithm'] == algorithm]['provider'])
        # Loop over providers, removing empty rows
        for prov in providers:
            # Pull out all rows with a certain provider
            cdf = sdf.xs(prov, level='provider').dropna(axis=1, how='all')
            # Create stats by algorithm, provider
            algorithm_stats_byprov[algorithm][prov] = create_stats(cdf)


    dfs['stats_all'] = df_allstats
    dfs['stats_byalgorithms'] = algorithm_stats_all
    dfs['stats_byprov'] = algorithm_stats_byprov


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

def get_fy_date_ranges(report_date: datetime.date) -> (datetime.date, datetime.date, datetime.date, datetime.date):
    """
    Returns four datetime.date() objects for start & end of financial year and quarter

    :param report_date: reference datetime for the date ranges
    :returns: y_start, y_end, q_start, q_end datetime.date() objects
    """
    with fiscal_calendar(start_month=7):
        y = FiscalDate(report_date.year, report_date.month, report_date.day)
        py = FiscalYear(y.fiscal_year)
        y_start = py.start.date()
        y_end = py.end.date()
        pq = FiscalQuarter(y.fiscal_year, y.fiscal_quarter)
        q_start = pq.start.date()
        q_end = pq.end.date()
        return y_start, y_end, q_start, q_end


def get_cnts(df_dict: dict[str, pd.DataFrame], all_provs: list, date_fieldname: str,
             start_date: datetime.date, end_date: datetime.date) -> (int, int):
    """
    Gets kilometres and borehole counts for a provider

    :param df_dict: source dataframe dict
    :param all_provs: provider string
    :param date_fieldname: name of date field to use in calculations
    :param start_date: start date, datetime.date() object
    :param end_date: start date, datetime.date() object
    :returns: tuple (borehole counts, borehole kilometres)
    """
    cnts = {}
    kms = {}
    for prov in all_provs:
        # Borehole counts and depths from 'start_date' to 'end_date'
        cnts[prov], kms[prov] = calc_bh_depths(df_dict, prov, date_fieldname, start_date, end_date, True)
    return cnts, kms

def calc_fyq(report_date: datetime.date, date_fieldname: str,  df_dict: dict[str, pd.DataFrame], all_provs: list):
    """
    Calculate quarterly and yearly data

    :param report_date: base date from which reporting occurs
    :param date_fieldname: name of date field to use in calculations
    :param df_dict: source dataframe dict
    :param all_provs: list of providers
    :returns: tuple of SimpleNamespace() fields are:
        start = start date
        end = end date
        kms_list = list of borehole kms in provider order
        cnts_list = list of borehole counts in provider order
    """

    y = SimpleNamespace()
    q = SimpleNamespace()

    # Get start and end of previous financial year and quarter
    y.start, y.end, q.start, q.end = get_fy_date_ranges(report_date)

    # Number of boreholes added and total amount of depths added from tne end of last FY relative to report_date
    y_cnts, y_kms = get_cnts(df_dict, all_provs, date_fieldname, y.start, y.end)

    # Number of boreholes added and total amount of depths added from the end of the quarter realtive to report_date
    q_cnts, q_kms = get_cnts(df_dict, all_provs, date_fieldname, q.start, q.end)

    # List of providers' kms, in provider order
    q.kms_list = [q_kms[prov] for prov in all_provs]
    y.kms_list = [y_kms[prov] for prov in all_provs]

    # List of providers' counts, in provider order
    q.cnt_list = [q_cnts[prov] for prov in all_provs]
    y.cnt_list = [y_cnts[prov] for prov in all_provs]

    return y, q


def assemble_report(report_file: str, report_date: datetime.date, date_fieldname: str, df_dict: dict[str:pd.DataFrame], plot_dir: str, prefix: str, brief: bool):
    """
    Generates a set of plot files and writes out PDF report

    :param report_file: output directory & file for report
    :param report_date: base date from which reporting occurs
    :param date_fieldname: name of date field to use in calculations
    :param df_dict: source dataframe dict, key is log type
    :param plot_dir: where plots are stored
    :param prefix: sorting prefix
    :param brief: if True will create a smaller report
    """
    # Remove old plots
    shutil.rmtree(plot_dir)
    os.mkdir(plot_dir)
    p = Plots(plot_dir)

    report = ReportTableData()
    # Check 'df_dict' for empty dataframes
    if not any(key in df_dict and ((type(df_dict[key]) is dict and df_dict[key] != {}) or (isinstance(df_dict[key], pd.DataFrame) and not df_dict[key].empty)) for key in ('log1','log2','empty','nodata')):
        print("Datasets are empty, please create them before enabling plots")
        print(f"df_dict.keys()={df_dict.keys()}")
        print(f"df_dict={df_dict}")
        sys.exit(1)
    df_all_list = [df_dict[key] for key in ('log1', 'log2', 'empty', 'nodata') if not df_dict[key].empty]
    df_all = pd.concat(df_all_list)
    df_log2_list = [df for df in (df_dict['log2'], df_dict['empty'][df_dict['empty']['log_type'] == '2']) if not df.empty]
    dfs_log2_all = pd.concat(df_log2_list)

    # Count boreholes up until the report_date
    all_provs, all_counts = np.unique(df_all[(df_all[date_fieldname] < report_date)].drop_duplicates(subset='nvcl_id')['provider'], return_counts=True)

    if not brief:
        # Count log1 data for all providers
        log1_provs, log1_counts = np.unique(df_dict['log1'].drop_duplicates(subset='nvcl_id')['provider'], return_counts=True)
        # Insert zeros for any providers that are missing
        if len(all_provs) > len(log1_provs):
            log1_counts, log1_provs = fill_in(all_provs, log1_counts, log1_provs)
        # Make log1 table
        report.add_table(list(log1_provs), list(log1_counts), "Log 1 Counts by Provider")

        # Count log2 data for all providers
        log2_provs, log2_counts = np.unique(df_dict['log2'].drop_duplicates(subset='nvcl_id')['provider'], return_counts=True)
        # Insert zeros for any providers that are missing
        if len(all_provs) > len(log2_provs):
            log2_counts, log2_provs = fill_in(all_provs, log2_counts, log2_provs)
        # Make log2 table
        report.add_table(list(log2_provs), list(log2_counts), "Log 2 Counts by Provider")

        # Count 'nodata' data for all providers
        nodata_provs, nodata_counts = np.unique(df_dict['nodata'].drop_duplicates(subset='nvcl_id')['provider'], return_counts=True)
        # Insert zeros for any provider that are missing
        if len(all_provs) > len(nodata_provs):
            nodata_counts, nodata_provs = fill_in(all_provs, nodata_counts, nodata_provs)

        # Make nodata table
        report.add_table(list(nodata_provs), list(nodata_counts), "'No data' Counts by Provider")

        # Count 'empty' data for all provider 
        df_empty_log1 = df_dict['empty'][df_dict['empty']['log_type'] == '1']
        empty_provs, empty_counts = np.unique(df_empty_log1.drop_duplicates(subset='nvcl_id')['provider'], return_counts=True)
        # Insert zeros for any providers that are missing
        if len(all_provs) > len(empty_provs):
            empty_counts, empty_provs = fill_in(all_provs, empty_counts, empty_provs)

        # Make empty counts table
        report.add_table(list(empty_provs), list(empty_counts), "'empty' Counts by Provider")

        # Plot percentage of boreholes by provider and data present
        p.plot_borehole_percent(nodata_counts, log1_counts, all_counts, log1_provs, nodata_provs, empty_provs)

    # Plot number of boreholes by provider
    report_date_str = report_date.strftime('%d/%m/%Y')
    p.plot_borehole_number(df_dict, all_provs, all_counts, f"Number of boreholes by Provider up to {report_date_str}", "borehole_number.png")

    # Table of number of boreholes by provider
    report.add_table(list(all_provs), list(all_counts), f"Number of boreholes by Provider up to {report_date_str}")

    # Calculate a list of total depths in kms, one value for each provider, up until the report date
    nkilometres_totals = [ calc_bh_depths(df_dict, prov, date_fieldname, end_date=report_date) for prov in all_provs ]

    # Make number of kilometres by provider table
    report.add_table(list(all_provs), nkilometres_totals, f"Number of borehole kilometres by Provider up to {report_date_str}")
    
    # Plot borehole kilometres by provider
    p.plot_borehole_kilometres(all_provs, nkilometres_totals, f"Number of borehole kilometres by provider up to {report_date_str}", "borehole_kilometres.png")

    # Calculate quarterly and financial year data
    y, q = calc_fyq(report_date, date_fieldname, df_dict, all_provs)

    # Pretty printed date strings
    y_start_date_str = y.start.strftime('%d/%m/%Y')
    q_start_date_str = q.start.strftime('%d/%m/%Y')
    y_end_date_str = y.end.strftime('%d/%m/%Y')
    q_end_date_str = q.end.strftime('%d/%m/%Y')
    y_end_date_pretty = y.end.strftime("%A %d %b %Y")
    q_end_date_pretty = q.end.strftime("%A %d %b %Y")

    # Plot yearly and quarterly comparisons for counts by provider
    p.plot_borehole_number(df_dict, all_provs, y.cnt_list, title=f"Borehole counts for this FY  ({y_start_date_str} - {y_end_date_str})", filename="borehole_number_y.png")
    p.plot_borehole_number(df_dict, all_provs, q.cnt_list, title=f"Borehole counts for this quarter ({q_start_date_str} - {y_end_date_str})", filename="borehole_number_q.png")

    # Tabulate yearly and quarterly comparisons for counts by provider
    report.add_table(list(all_provs), q.cnt_list, f"Borehole counts by Provider for this quarter ({q_start_date_str} - {q_end_date_str})")
    report.add_table(list(all_provs), y.cnt_list, f"Borehole counts by Provider for this FY ({y_start_date_str} - {y_end_date_str})")

    # Plot yearly and quarterly comparisons for kilometres by provider
    p.plot_borehole_kilometres(all_provs, y.kms_list, title=f"Borehole kilometres for this FY ({y_start_date_str} - {y_end_date_str})", filename="borehole_kilometres_y.png")
    p.plot_borehole_kilometres(all_provs, q.kms_list, title=f"Borehole kilometres for this quarter ({q_start_date_str} - {q_end_date_str})", filename="borehole_kilometres_q.png")
 
    # Tabulate yearly and quarterly comparisons for kilometres by provider
    report.add_table(list(all_provs), q.kms_list, f"Borehole kilometres by Provider for this quarter ({q_start_date_str} - {q_end_date_str})")
    report.add_table(list(all_provs), y.kms_list, f"Borehole kilometres by Provider for this FY ({y_start_date_str} - {y_end_date_str})")

    # Plot word clouds
    # plot_wordclouds(dfs_log2_all)

    if not brief:

        # Get algorithms from any available service
        # Create a dict mapping from 
        algoid2ver = {}
        """for prov in PROV_LIST[2:]:
            param = param_builder(prov, max_boreholes=1)
            if not param:
                print(f"Cannot build parameters for {prov}: {param}")
                continue

            print(f"{param=}")
            reader = NVCLReader(param)
            if not reader.wfs:
                print(f"ERROR! Cannot connect to {prov}")
                continue
            algoid2ver = reader.get_algorithms()
            print(f"{algoid2ver=}")
            if len(algoid2ver.keys()) > 0:
                break
        """
        algoid2ver={'7': '500', '20': '600', '26': '601', '82': '703', '96': '704', '101': '704', '25': '600', '31': '601', '87': '703', '9': '500', '22': '600', '28': '601', '84': '703', '98': '704', '10': '500', '23': '600', '29': '601', '85': '703', '99': '704', '11': '500', '24': '600', '30': '601', '86': '703', '100': '704', '21': '600', '27': '601', '83': '703', '97': '704', '8': '500', '40': '631', '32': '630', '1': '500', '12': '600', '48': '700', '74': '703', '88': '704', '108': '705', '41': '631', '33': '630', '2': '500', '13': '600', '49': '700', '75': '703', '89': '704', '109': '705', '42': '631', '34': '630', '3': '500', '14': '600', '50': '700', '76': '703', '90': '704', '110': '705', '43': '631', '35': '630', '4': '500', '15': '600', '51': '700', '77': '703', '91': '704', '111': '705', '44': '631', '36': '630', '5': '500', '16': '600', '52': '700', '78': '703', '92': '704', '112': '705', '45': '631', '37': '630', '17': '600', '53': '700', '79': '703', '93': '704', '113': '705', '46': '631', '38': '630', '18': '600', '54': '700', '80': '703', '94': '704', '114': '705', '47': '631', '39': '630', '19': '600', '55': '700', '81': '703', '95': '704', '115': '705', '6': '500', '116': '705', '117': '705', '56': '701', '62': '702', '68': '703', '142': '708', '134': '707', '102': '704', '118': '705', '126': '706', '133': '706', '149': '708', '141': '707', '125': '705', '58': '701', '64': '702', '70': '703', '144': '708', '136': '707', '104': '704', '120': '705', '128': '706', '59': '701', '65': '702', '71': '703', '145': '708', '137': '707', '105': '704', '121': '705', '129': '706', '60': '701', '66': '702', '72': '703', '146': '708', '138': '707', '106': '704', '122': '705', '130': '706', '61': '701', '67': '702', '73': '703', '147': '708', '139': '707', '107': '704', '123': '705', '131': '706', '63': '702', '69': '703', '143': '708', '135': '707', '103': '704', '119': '705', '127': '706', '57': '701', '148': '708', '140': '707', '124': '705', '132': '706'}
        algoid2ver['0'] = '0'

        # Plot algorithms
        p.plot_algorithms(df_dict, algoid2ver)

        # Plot uTSAS graphs
        p.plot_spectrum_group(df_dict, algoid2ver, prefix)

        '''
        pd.DataFrame({'algo':grp_algos, 'counts':grp_counts}).plot.bar(x='algo',y='counts', rot=20)
        [min_algos, min_counts] = np.unique(dfs['log1'][dfs['log1'].algorithm.str.startswith('Min')]['algorithm'], return_counts=True)
        pd.DataFrame({'algo':min_algos, 'counts':min_counts}).plot.bar(x='algo',y='counts', rot=20)
        index = [x.replace('Grp','') for x in grp_algos]
        df = pd.DataFrame({'Grp': grp_counts, 'Min': min_counts}, index=index).plot.bar(rot=20)
        '''
        # Plot element graphs
        p.plot_elements(dfs_log2_all)

        # Plot geophysics
        p.plot_geophysics(dfs_log2_all)

    metadata = { "Authors": "Vincent Fazio & Shane Mule",
                 "Sources": "This report was compiled from NVCL datasets downloaded from the NVCL nodes managed\nby the geological surveys of QLD, NSW, Vic, Tas, NT, SA & WA", 
                 "Base Report Date": report_date.strftime("%A %b %d %Y"),
                 "Using quarterly data ending at": f"{q_end_date_pretty}",
                 "Using EOFY data ending at": f"{y_end_date_pretty}"
    }

    # Finally write out PDF report
    date_str = report_date.strftime("%Y-%m-%d")
    if report_file is None:
        report_filename = f"{date_str}-report-brief.pdf" if brief else f"{date_str}-report.pdf"
    else:
        report_filename = report_file
    write_report(report_filename, plot_dir, report, metadata, p, brief)


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
    ca_stats = pd.concat([ca_stats, nbores])
    nmetres = cdf.loc['mincnts']
    if isinstance(nmetres, pd.DataFrame):
        nmetres = nmetres.sum()
    nmetres.name = 'mincnts'
    ca_stats = pd.concat([ca_stats, nmetres])
    IDs = np.unique(list(filter(lambda x: re.match(r"algorithm_\d+", x), cdf.index.tolist())))
    for ID in IDs:
        cID = cdf.loc[ID]
        if isinstance(cID, pd.DataFrame):
            cID = cID.sum()
        cID.name = ID
        ca_stats = pd.concat([ca_stats, cID])
    # Add up duplicates
    ca_stats = ca_stats.groupby(ca_stats.index).sum()
    
    return pd.DataFrame(ca_stats).transpose()
