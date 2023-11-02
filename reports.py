
# Python imports
import re
import datetime
import shutil
import os
import sys

# External imports
import pandas as pd
pd.options.mode.chained_assignment = None

import numpy as np
from itertools import zip_longest
from nvcl_kit.reader import NVCLReader
from nvcl_kit.param_builder import param_builder
from fiscalyear import FiscalDate

# Local imports
from make_plots import (plot_borehole_percent, plot_borehole_number, plot_borehole_kilometres,
                          plot_geophysics, plot_elements, plot_spectrum_group, plot_algorithms)
from make_pdf import write_report
from constants import PROV_LIST


def make_table(table_data: list, title_list: list, table_header: list, table_datarows: list, title: str):
        """ Makes a table data structure for passing to report building routines

        :param table_data: list passed to report building routines
        :param title_list: list of table titles
        :param table_header: list of table header strings
        :param table_datarow: list of data rows to be inserted into table, each row same length as headers string list
        :param title: title of data table
        """
        table_rows = [table_header]
        table_rows.append(table_datarows)
        table_data.append(table_rows)
        title_list.append(title)

def calc_metric_diffs(all_keys: list, larger: dict, smaller: dict):
    '''
    Calculate numerical difference between dicts of metrics, key is provider name
    NB: if value is missing zero is returned

    :param all_keys: list of all possible dict keys (provider names)
    :param larger: dict of metric values keys are provider names
    :param smaller: dict of metric values keys are provider names
    :returns: array of differences, one for each value in 'all_keys'
    '''
    # print(f"{all_keys=} {larger=} {smaller=}")
    result = []
    for key in all_keys:
        if key in larger and key in smaller and larger[key] > smaller[key]:
            result.append(larger[key] - smaller[key])
        else:
            result.append(0)
    # print(f"{result=}")
    return result    


def calc_bh_kms(dfs: dict[str:pd.DataFrame], prov: str, start_date: datetime.date = None,
                end_date: datetime.date = None, return_cnts: bool = False) -> int:
    """
    Assemble a dict of borehole depths

    :param dfs: source dataframe dict
    :param prov: name of data provider, state or territory
    :param start_date: optional start date, datetime.date() object
    :param end_date: optional end date, datetime.date() object
    :param return_cnts: if true will return counts
    :returns: (optional) borehole counts, sum of borehole depths (units: kilometres)
    """
    # Filter by data provider, optional date range
    print(f"calc_bh_kms({prov})")
    if dfs['log1'].empty:
        if not return_cnts:
            return 0
        return 0, 0

    if start_date is None or end_date is None:
        df = dfs['log1'][dfs['log1']['provider'] == prov]
    else:
        df = dfs['log1'][(dfs['log1']['provider'] == prov) & (dfs['log1']['modified_datetime'] > start_date) & (dfs['log1']['modified_datetime'] < end_date)]
    bh_kms = {}
    nvcl_ids = np.unique(df['nvcl_id'])
    cnts = len(nvcl_ids)
    for nvcl_id in nvcl_ids:
        depth_set = set()
        # Filter by NVCL id
        nvclid_df = df[df.nvcl_id == nvcl_id]
        for idx, row in nvclid_df.iterrows():
            depths = [depth for depth, minerals in row['data']]
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
    print(f"Calculating initial statistics ...{dfs}")
    # Loop around for each provider
    for prov in prov_list:
        cdf = dfs['log1'][dfs['log1']['provider'] == prov]
        if cdf.empty:
            continue

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
                # print(IDs[iC])
                algo = f"algorithm_{IDs[iC]}"
                algos.append(algo)
                if algo not in df_cstats:
                    df_cstats.insert(0, algo, 0 * len(df_cstats))
                df_cstats[algo].loc[col.dropna().tolist()] += 1
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


def get_cnts(dfs: dict[str, pd.DataFrame], all_provs: list,
             start_date: datetime.date, end_date: datetime.date) -> (int, int):
    """
    Gets kilometres and borehole counts for a provider

    :param dfs: source dataframe dict
    :param prov: provider string
    :param start_date: start date, datetime.date() object
    :param end_date: start date, datetime.date() object
    :returns: tuple (borehole counts, borehole kilometres)
    """
    cnts = {}
    kms = {}
    for prov in all_provs:
        cnts[prov], kms[prov] = calc_bh_kms(dfs, prov, start_date, end_date, True)
    return cnts, kms


def plot_results(dfs: dict[str:pd.DataFrame], plot_dir: str, prefix: str, brief: bool):
    """
    Generates a set of plot files

    :param dfs: source dataframe dict
    :param prefix: sorting prefix
    :param plot_dir: where plots are stored
    :param brief: if True will create a smaller report
    """
    # Remove old plots
    shutil.rmtree(plot_dir)
    os.mkdir(plot_dir)

    table_data = []
    title_list = []
    if not any(key in dfs and ((type(dfs[key]) is dict and dfs[key] != {}) or (isinstance(dfs[key], pd.DataFrame) and not dfs[key].empty)) for key in ('log1','log2','empty','nodata')):
        print("Datasets are empty, please create them before enabling plots")
        print(f"dfs.keys()={dfs.keys()}")
        print(f"dfs={dfs}")
        sys.exit(1)
    df_all = pd.concat([dfs['log1'], dfs['log2'], dfs['empty'], dfs['nodata']])
    dfs_log2_all = pd.concat([dfs['log2'], dfs['empty'][dfs['empty']['log_type'] == '2']])
    all_provs, all_counts = np.unique(df_all.drop_duplicates(subset='nvcl_id')['provider'], return_counts=True)
    print(f"{all_provs=}")
    print(f"{all_counts=}")

    if not brief:
        # Count log1 data for all providers
        log1_provs, log1_counts = np.unique(dfs['log1'].drop_duplicates(subset='nvcl_id')['provider'], return_counts=True)
        # Insert zeros for any providers that are missing
        if len(all_provs) > len(log1_provs):
            log1_counts, log1_provs = fill_in(all_provs, log1_counts, log1_provs)
        # Make log1 table
        make_table(table_data, title_list, list(log1_provs), list(log1_counts), "Log 1 Counts by Provider")

        # Count log2 data for all providers
        log2_provs, log2_counts = np.unique(dfs['log2'].drop_duplicates(subset='nvcl_id')['provider'], return_counts=True)
        # Insert zeros for any providers that are missing
        if len(all_provs) > len(log2_provs):
            log2_counts, log2_provs = fill_in(all_provs, log2_counts, log2_provs)
        # Make log2 table
        make_table(table_data, title_list, list(log2_provs), list(log2_counts), "Log 2 Counts by Provider")

        # Count 'nodata' data for all providers
        nodata_provs, nodata_counts = np.unique(dfs['nodata'].drop_duplicates(subset='nvcl_id')['provider'], return_counts=True)
        # Insert zeros for any provider that are missing
        if len(all_provs) > len(nodata_provs):
            nodata_counts, nodata_provs = fill_in(all_provs, nodata_counts, nodata_provs)

        # Make nodata table
        make_table(table_data, title_list, list(nodata_provs), list(nodata_counts), "'No data' Counts by Provider")

        # Count 'empty' data for all provider 
        df_empty_log1 = dfs['empty'][dfs['empty']['log_type'] == '1']
        empty_provs, empty_counts = np.unique(df_empty_log1.drop_duplicates(subset='nvcl_id')['provider'], return_counts=True)
        # Insert zeros for any providers that are missing
        if len(all_provs) > len(empty_provs):
            empty_counts, empty_provs = fill_in(all_provs, empty_counts, empty_provs)

        # Make empty counts table
        make_table(table_data, title_list, list(empty_provs), list(empty_counts), "'empty' Counts by Provider")

        # Plot percentage of boreholes by provider and data present
        plot_borehole_percent(plot_dir, nodata_counts, log1_counts, all_counts, log1_provs, nodata_provs, empty_provs)

    # Plot number of boreholes by provider
    plot_borehole_number(dfs, plot_dir, all_provs, all_counts)
    
    # Table of number of boreholes by provider
    make_table(table_data, title_list, list(all_provs), list(all_counts), "Number of boreholes by Provider")

    # Calculate a list of number of kilometres, one value for each provider 
    nkilometres_totals = [ calc_bh_kms(dfs, prov) for prov in all_provs ]

    # Make number of kilometres by provider table
    make_table(table_data, title_list, list(all_provs), nkilometres_totals, "Number of borehole kilometres by Provider")
    
    # Plot borehole kilometres by provider
    plot_borehole_kilometres(all_provs, nkilometres_totals, plot_dir)

    # Calculate borehole counts and kilometres
    y_start, y_end, q_start, q_end = get_fy_date_ranges()
    y_cnts, y_kilometres = get_cnts(dfs, all_provs, y_start, y_end)
    q_cnts, q_kilometres = get_cnts(dfs, all_provs, q_start, q_end)

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
    plot_borehole_number(dfs, plot_dir, all_keys, y_diffs, title=f"Borehole counts since end of last financial year {y_date}", filename="borehole_number_y.png")
    plot_borehole_number(dfs, plot_dir, all_keys, q_diffs, title=f"Borehole counts since last quarter {q_date}", filename="borehole_number_q.png")

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
    print(f"{all_keys=}")
    print(f"{nkilometres_dict=}")
    plot_borehole_kilometres(all_keys, y_diffs, plot_dir, title=f"Borehole kilometres since end of last financial year {y_date}", filename="borehole_kilometres_y.png")
    plot_borehole_kilometres(all_keys, q_diffs, plot_dir, title=f"Borehole kilometres since last quarter {q_date}", filename="borehole_kilometres_q.png")
 
    # Tabulate yearly and quarterly comparisons for kilometres by provider
    make_table(table_data, title_list, list(all_keys), q_diffs, f"Number of borehole kilometres by Provider since last quarter {q_date}")
    make_table(table_data, title_list, list(all_keys), y_diffs, f"Number of borehole kilometres by Provider since end of last financial year {y_date}")

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
        plot_algorithms(dfs, plot_dir, algoid2ver)

        # Plot uTSAS graphs
        plot_spectrum_group(dfs, plot_dir, algoid2ver, prefix)

        '''
        pd.DataFrame({'algo':grp_algos, 'counts':grp_counts}).plot.bar(x='algo',y='counts', rot=20)
        [min_algos, min_counts] = np.unique(dfs['log1'][dfs['log1'].algorithm.str.startswith('Min')]['algorithm'], return_counts=True)
        pd.DataFrame({'algo':min_algos, 'counts':min_counts}).plot.bar(x='algo',y='counts', rot=20)
        index = [x.replace('Grp','') for x in grp_algos]
        df = pd.DataFrame({'Grp': grp_counts, 'Min': min_counts}, index=index).plot.bar(rot=20)
        '''
        # Plot element graphs
        plot_elements(dfs_log2_all, plot_dir)

        # Plot geophysics
        plot_geophysics(dfs_log2_all, plot_dir)

    now = datetime.datetime.now()
    metadata = { "Authors": "Vincent Fazio & Shane Mule",
                 "Sources": "This report was compiled from NVCL datasets downloaded from the NVCL nodes managed\nby the state geological surveys of QLD, NSW, Vic, Tas, NT, SA & WA", 
                 "Report Date": now.strftime("%A %b %d %Y"),
                 "Using quarterly data from": q_date_pretty,
                 "Using EOFY data from": y_date_pretty }

    # Finally write out pdf report
    if brief:
        write_report("report-brief.pdf", plot_dir, table_data, title_list, metadata, brief)
    else:
        write_report("report.pdf", plot_dir, table_data, title_list, metadata, brief)


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
    print(f"1:{ca_stats=}")
    nmetres = cdf.loc['mincnts']
    if isinstance(nmetres, pd.DataFrame):
        nmetres = nmetres.sum()
    nmetres.name = 'mincnts'
    ca_stats = pd.concat([ca_stats, nmetres])
    print(f"2:{ca_stats=}")
    IDs = np.unique(list(filter(lambda x: re.match(r"algorithm_\d+", x), cdf.index.tolist())))
    for ID in IDs:
        cID = cdf.loc[ID]
        if isinstance(cID, pd.DataFrame):
            cID = cID.sum()
        cID.name = ID
        ca_stats = pd.concat([ca_stats, cID])
    print(f"3:{ca_stats=}")
    print(f"{ca_stats.index=}")
    print(f"Before {ca_stats.index.duplicated().any()=}")
    # Add up duplicates
    ca_stats = ca_stats.groupby(ca_stats.index).sum()
    print(f"After {ca_stats.index.duplicated().any()=}")
    
    return pd.DataFrame(ca_stats).transpose()
