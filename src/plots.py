import re
import os

from wordcloud import WordCloud, STOPWORDS
from periodictable import elements
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# Matplotlib legend positioning constant
BBX2A = (1.0, 0.5)

"""
Functions used for plotting in reports
"""

def sort_cols(df, prefix='version', split_tok='_'):
    """ Sort columns by value in pandas dataframe
    Column names are assumed to be in format '<prefix><split_tok><value>'

    :param df: pandas dataframe
    :param prefix: optional column prefix to search for in column names, default value is 'version'
    :param split_tok: optional split token in column names, default value is '_'
    :returns: list of sorted & transformed columns
    """
    cols = sorted(df.columns)
    anums = []
    for c in cols:
        if (re.match('^' + prefix, c)):
            anums.append(c.split(split_tok)[1])
    return [prefix + split_tok + str(x) for x in sorted([int(x) for x in anums])]


def plot_borehole_percent(plot_dir, nodata_counts, log1_counts, all_counts, log1_provs, nodata_provs, empty_provs):
    # Plot percentage of boreholes by provider and data present
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1)
    log1_rel = [i / j * 100 for i, j in zip(log1_counts, all_counts)]
    ax.bar(log1_provs, log1_rel, label='HyLogger data')
    nodata_rel = [i / j * 100 for i, j in zip(nodata_counts, all_counts)]
    ax.bar(nodata_provs, nodata_rel, bottom=log1_rel, label="No HyLogger data")
    empty = all_counts-(log1_counts + nodata_counts)
    empty_rel = [i / j * 100 for i, j in zip(empty, all_counts)]
    ax.bar(empty_provs, empty_rel, bottom=[i+j for i,j in zip(log1_rel, nodata_rel)], label="No data")
    plt.ylabel("Percentage of boreholes (%)")
    plt.title("Percentage of boreholes by provider and data present")
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(plot_dir, "borehole_percent.png"))

def plot_borehole_number(dfs: dict, plot_dir: str, all_provs: np.array, all_counts: np.array, title: str, filename: str):
    """ Plot number of boreholes by provider

    :param dfs: dataframe dict, key is log type
    :param plot_dir: directory in filesystem where plot is saved
    :param all_provs: array of provider strings
    :param all_counts: array of counts
    :param title: plot title
    :param filename: plot filename
    """
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax1 = ax.bar(all_provs, all_counts)
    for r1 in ax1:
        h1 = r1.get_height()
        plt.text(r1.get_x() + r1.get_width() / 2., h1, f"{h1}", ha='center', va='bottom', fontweight='bold')
    plt.ylabel("Number of boreholes")
    plt.title(title)
    plt.savefig(os.path.join(plot_dir, filename))

    # Plot number of boreholes for geology by provider
    dfs_log1_geology = dfs['log1'][dfs['log1']['algorithm'].str.contains(r'^(?:Strat|Form|Lith)', case=False)]
    if not dfs_log1_geology.empty:
        ax = dfs_log1_geology.drop_duplicates('nvcl_id').groupby(['provider', 'algorithm']).size().unstack().plot(kind='bar', rot=0, figsize=(30, 15), title="Number of boreholes for geology by provider")
        ax.set(xlabel='Provider', ylabel="Number of boreholes")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "log1_geology.png"))
        df = dfs_log1_geology.drop_duplicates('nvcl_id').groupby(['provider', 'algorithm']).size().unstack()
        return df.to_numpy().tolist()

def plot_borehole_kilometres(all_provs: np.array, all_counts: np.array, plot_dir: str, title: str, filename:str):
    """
    Plot number of borehole kilometres by provider

    :param all_provs: array of provider strings
    :param all_counts: array of counts
    :param plot_dir: directory in filesystem where plot is saved
    :param title: plot title
    :param filename: plot filename
    """
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1)
    #print(f"{all_provs=}")
    #print(f"{all_counts=}")
    ax1 = ax.bar(all_provs, all_counts)
    for r1 in ax1:
        h1 = r1.get_height()
        if isinstance(h1, float):
            plt.text(r1.get_x() + r1.get_width() / 2., h1, f"{h1:.1f}", ha='center', va='bottom', fontweight='bold')
        else:
            plt.text(r1.get_x() + r1.get_width() / 2., h1, f"{h1}", ha='center', va='bottom', fontweight='bold')
    plt.ylabel("Number of borehole kilometres")
    plt.title(title)
    plt.savefig(os.path.join(plot_dir, filename))


def plot_wordclouds(dfs_log2_all: pd.DataFrame, plot_dir: str):
    """
    Plot word clouds

    :param dfs_log2_all: dataframe to plot
    :param plot_dir: directory in filesystem where plot is saved
    """
    words = [x.replace(' ','_') for x in list(dfs_log2_all['algorithm'])]
    wordcloud = WordCloud(width=1600, height=800, background_color="white",
                          stopwords = STOPWORDS, max_words=200).generate(' '.join(words))
    plt.figure(figsize=(20, 10))
    plt.imshow(wordcloud, interpolation = 'bilinear')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(plot_dir, 'log2_wordcloud.png'))

    words = [x.replace(' ','_') for x in list(dfs_log2_all['algorithm'])]
    wordcloud = WordCloud(width=1600, height=800, background_color="white",
                          stopwords = STOPWORDS, max_words=200).generate(' '.join(words))
    plt.figure(figsize=(20, 10))
    plt.imshow(wordcloud, interpolation = 'bilinear')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(plot_dir, 'log1_wordcloud.png'))


def plot_geophysics(dfs_log2_all, plot_dir):
    """
    Plot geophysics data by provider

    :param dfs_log2_all: dataframe to plot
    :param plot_dir: directory in filesystem where plot is saved
    """
    phys_include = ['magsus', 'mag sus', 'cond']
    df_phys = dfs_log2_all[(dfs_log2_all['algorithm'].str.contains(('|'.join(phys_include)), case=False))]
    if not df_phys.empty:
        ax = df_phys.drop_duplicates('nvcl_id')['provider'].value_counts().plot(kind='bar',
                                                                                rot=0, figsize=(10, 10),
                                                                                title="Geophysics data by provider")
        ax.set(xlabel='provider', ylabel="Number of boreholes")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "geophys_prov.png"))
        plt.close('all')

        # Plot geophysics
        ax = df_phys['algorithm'].value_counts().plot(kind='bar',
                                                      figsize=(10, 10), rot=90, title='Geophysics')
        ax.set(xlabel='data', ylabel="Number of data records")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "geophys_count.png"))
        plt.close('all')



def plot_elements(dfs_log2_all, plot_dir):
    """
    Plot elements graphs

    :param dfs_log2_all: dataframe to plot
    :param plot_dir: directory in filesystem where plot is saved
    """
    # el_exclude = np.concatenate((suffixes, ["white mica", "featex", "mWt", "Chlorite Epidote Index", "albedo", "Core colour from imagery", "ISM", "Kaolinite Crystallinity", "Mica", "smooth", "TIR", "magsus", "mag sus", "cond", "reading", "pfit", "color"]))
    # df_excluded = dfs_log2_all[~(dfs_log2_all['algorithm'].str.contains(('|'.join(el_exclude)), case=False))]
    df_log2_el = dfs_log2_all[(dfs_log2_all['algorithm'].str.contains(('|'.join(["ppm", "pct", "%", "per", "arsen", "Sillimanite", "PbZn"])), case=False))]
    df_log2_el = pd.concat([df_log2_el, dfs_log2_all[(dfs_log2_all['algorithm'].str.match("^(" + '|'.join([str(x) for x in list(elements)]) + ")$", case=False))]])
    df_log2_el = pd.concat([df_log2_el, dfs_log2_all[(dfs_log2_all['algorithm'].str.match("^(" + '|'.join([str(x) for x in list(elements)]) + r")\s+.*$", case=False))]])
    # df_diff = df_excluded[~df_excluded.apply(tuple,1).isin(df_log2_el.apply(tuple, 1))]
    df_log2_el['element'] = df_log2_el['algorithm'].replace({'(?i)ppm|(?i)pct|(?i)per': ''}, regex=True).replace({'([A-Za-z0-9]+)(_| )*(.*)': r'\1'}, regex=True).replace({r'(\w+)(0|1|SFA)': r'\1'}, regex=True)
    df_log2_el['suffix'] = [a.replace(b, '') for a, b in zip(df_log2_el['algorithm'], df_log2_el['element'])]
    df_log2_el['suffix'] = df_log2_el['suffix'].replace({'^(_.*)': r' \1'}, regex=True)
    df_log2_el['element'] = df_log2_el['element'].replace({'(?i)Arsen$': 'Arsenic'}, regex=True).apply(lambda x: (x[0].upper() + x[1].lower() + x[2:]) if len(x) > 2 else x[0].upper()+x[1].lower() if len(x) > 1 else x[0].upper())

    # Plot element data by provider
    if not df_log2_el.empty:
        ax = df_log2_el.drop_duplicates('nvcl_id')['provider'].value_counts().plot(kind='bar', rot=0, figsize=(10, 10), title="Element data by provider")
        ax.set(xlabel='provider', ylabel="Number of boreholes")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "elems_prov.png"))
        plt.close()
        ax = df_log2_el['element'].value_counts()
        if not ax.empty:
            p = ax.plot(kind='bar', figsize=(40, 20), title='Elements')
            p.set(xlabel='Element', ylabel="Number of data records")
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "elems_count.png"))
            plt.close()

        # Plot element suffixes sorted by element
        ax = df_log2_el.groupby(['element', 'suffix']).size().unstack()
        if not ax.empty:
            p = ax.plot(kind='barh', stacked=False, figsize=(30, 50), title="Element suffixes sorted by element")
            p.set(xlabel='Element', ylabel="Number of data records")
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "elems_suffix.png"))
            plt.close()

        # Plot element suffixes for Sulfur
        ax = df_log2_el[df_log2_el['element'] == 'S'].groupby(['element', 'suffix']).size().unstack()
        if not ax.empty:
            p = ax.plot(kind='bar', stacked=False, rot=0, figsize=(35, 15), title="Element suffixes for Sulfur")
            p.set(xlabel='Element', ylabel="Number of data records")
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "elem_S.png"))
            plt.close()

        # Plot element suffixes
        ax = df_log2_el['suffix'].value_counts()
        if not ax.empty:
            p = ax.plot(kind='barh', figsize=(30, 50), title="Element suffixes")
            p.set(ylabel="Element suffix", xlabel="Number of data records")
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "elem_suffix_stats.png"))
            plt.close()

def plot_spectrum_group(dfs, plot_dir, algoid2ver, prefix):
    """
    Plot spectrum group

    :param dfs: dataframe to plot
    :param plot_dir: directory in filesystem where plot is saved
    :param algoid2ver: dictionary of algorithm id -> version id
    """
    # Grp=uTSAS[uTSAS.algorithm.str.startswith('Grp')]
    # [grp_algos, grp_counts] = np.unique(dfs['log1'][dfs['log1'].algorithm.str.startswith('Grp')]['algorithm'], return_counts=True)

    Grp_uTSAS = pd.DataFrame()
    for key, value in dfs['stats_byalgorithms'].items():
        if key.startswith('Grp') and key.endswith('uTSAS'):
            Grp_uTSAS = pd.concat([Grp_uTSAS, value], ignore_index=True).groupby(level=0).sum()
    Grp_uTSAS = dfcol_algoid2ver(Grp_uTSAS, algoid2ver)
    Min_uTSAS = pd.DataFrame()
    for key, value in dfs['stats_byalgorithms'].items():
        if key.startswith('Min') and key.endswith('uTSAS'):
            Min_uTSAS = pd.concat([Min_uTSAS, value]).groupby(level=0).sum()
    Min_uTSAS = dfcol_algoid2ver(Min_uTSAS, algoid2ver)

    # Plot Grp_uTSAS sorted by group name and version
    plot_df = Grp_uTSAS[sort_cols(Grp_uTSAS, prefix)]
    if not plot_df.empty:
        ax = plot_df.plot(kind='bar', figsize=(30, 10), title="Grp_uTSAS sorted by group name and version")
        ax.set(xlabel='Group', ylabel="Number of data records")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "grp_utsas.png"))

    # TODO: Standardise names before inserted into DataFrame
    plot_df = plot_df.loc[plot_df.index.intersection(['Carbonates', 'CARBONATE'])]
    if not plot_df.empty:
        ax = plot_df.plot(kind='barh', figsize=(20, 10), title="Grp_uTSAS sorted by group name and version")
        ax.set(ylabel='Group', xlabel="Number of data records")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "grp_utsas_carbonate.png"))

    plot_df = Min_uTSAS[sort_cols(Min_uTSAS, prefix)]
    if not plot_df.empty:
        ax = plot_df.plot(kind='bar', figsize=(30, 10), title="Min_uTSAS sorted by group name and version")
        ax.set(xlabel='Mineral', ylabel="Number of data records")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "min_utsas.png"))

    # Plot Min_uTSAS sorted by group name and version
    plot_df = Min_uTSAS[sort_cols(Min_uTSAS, prefix)]
    # TODO: Standardise names before inserted into DataFrame
    plot_df = plot_df.loc[plot_df.index.intersection(["Illitic Muscovite", "Muscovitic Illite", "MuscoviticIllite"])]
    if not plot_df.empty:
        ax = plot_df.plot(kind='barh', figsize=(20, 10), title="Min_uTSAS sorted by group name and version")
        ax.set(ylabel='Group', xlabel="Number of data records")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "min_utsas_misc.png"))
    plt.close('all')


def plot_algorithms(dfs, plot_dir, algoid2ver, prefix='version'):
    """
    Create graphs for algorithm reports

    :param dfs: dataframe to plot
    :param plot_dir: directory in filesystem where plot is saved
    :param algoid2ver: dictionary of algorithm id -> version id
    """
    algos = np.unique(dfs['log1'][dfs['log1'].algorithm.str.contains('^Min\d |Grp\d ')]['algorithm'])
    suffixes = np.unique([x.split()[1] for x in algos])
    dfs['log1']['versions'] = dfs['log1'].apply(lambda row: algoid2ver.get(row['algorithm_id'], '0'), axis=1)

    df_algo_stats = pd.DataFrame()
    df_algoID_stats = pd.DataFrame()
    for suffix in suffixes:
        provs, count = np.unique(dfs['log1'][dfs['log1'].algorithm.str.endswith(suffix)].drop_duplicates('nvcl_id')['provider'], return_counts=True)
        df_algo_stats = pd.concat([df_algo_stats, pd.DataFrame({suffix: count}, index=provs)], axis=1, sort=False)
        IDs, count = np.unique(dfs['log1'][dfs['log1'].algorithm.str.endswith(suffix)]['versions'], return_counts=True)
        # IDs = ['algorithm_'+x for x in IDs]
        vers = ['version_' + x for x in IDs]
        df_algoID_stats = pd.concat([df_algoID_stats, pd.DataFrame([np.array(count)], columns=vers, index=[suffix])], sort=False)

    # Plot number of boreholes for non-standard algorithms by provider
    dfs_log1_nonstd = dfs['log1'][~(dfs['log1']['algorithm'].str.contains((r'^(?:Grp|Min|Sample|Lith|HoleID|Strat|Form)'), case=False))]
    if not dfs_log1_nonstd.empty:
        dfs_log1_nonstd['Algorithm Prefix'] = dfs_log1_nonstd['algorithm'].replace({'(grp_|min_)': ''}, regex=True).replace({r'_*\d+$': ''}, regex=True)
        ax = dfs_log1_nonstd.drop_duplicates('nvcl_id').groupby(['provider', "Algorithm Prefix"]).size().unstack().plot(kind='bar', rot=0, figsize=(20, 10), title="Number of boreholes for non-standard algorithms by provider")
        ax.set(xlabel='Provider', ylabel="Number of boreholes")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "log1_nonstdalgos.png"))
    else:
        print("WARNING: There is insufficient data to produce non-standard Log1 algorithm stats")

    # Plot number of boreholes by algorithm and provider
    ax = df_algo_stats.plot(kind='bar', stacked=False, figsize=(20, 10), rot=0, title="Number of boreholes by algorithm and provider")
    ax.set(ylabel="Number of boreholes")
    # for p in ax.patches:
    #    ax.annotate(str(int(p.get_height())), (p.get_x()+p.get_width()/2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords="offset points", size=4, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "log1_algos.png"))

    # Plot number of data records of standard algorithms by version
    ax = df_algoID_stats[sort_cols(df_algoID_stats, prefix)].plot(kind='bar', stacked=False, figsize=(30, 10), rot=0, title="Number of data records of standard algorithms by version")
    ax.legend(loc="center left", bbox_to_anchor=BBX2A)
    # ax.grid(True, which='major', linestyle='-')
    # ax.grid(True, which='minor', linestyle='--')
    ax.set(ylabel="Number of data records")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "log1_algoIDs.png"))

    # Plot number of data records of standard algorithms by version and provider
    ax = dfs['log1'].groupby(['provider', 'versions']).size().unstack().plot(kind='bar', stacked=False, figsize=(30, 10), rot=0, title="Number of data records of standard algorithms by version and provider")
    ax.legend(loc='center left', bbox_to_anchor=BBX2A)
    ax.set(xlabel='Provider', ylabel="Number of data records")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "log1_algoIDs_prov.png"))
    plt.close('all')

    # Plot number of data records of algorithmXXX by version and provider
    for alg in df_algo_stats.columns:
        cAlg = dfs['log1'][dfs['log1'].algorithm.str.endswith(alg)]
        ax = cAlg.drop_duplicates('nvcl_id').groupby(['provider', 'versions']).size().unstack().plot(kind='bar', stacked=False, figsize=(30, 10), rot=0, title=f"Number of data records of {alg} by version and provider")
        ax.legend(loc="center left", bbox_to_anchor=BBX2A)
        ax.set(xlabel='Provider', ylabel="Number of boreholes")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"log1_{alg}-IDs_prov.png"))
    plt.close('all')


def dfcol_algoid2ver(df, algoid2ver):
        """ Renames columns in dataframe from algorithm id to version id

        :param df: pandas dataframe
        :param algoid2ver: dictionary of algorithm id -> version id
        :returns: transformed dataframe
        """
        df = df.rename({y: re.sub(r'algorithm_(\d+)', lambda x: f"version_{algoid2ver.get(x.group(1), '0')}", y) for y in df.columns}, axis='columns')
        df = df.groupby(level=0).sum()
        return df

