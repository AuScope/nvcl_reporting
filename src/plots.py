import re
import os

from wordcloud import WordCloud, STOPWORDS
from periodictable import elements
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker


# Matplotlib legend positioning constant
# Places legend outside graph on right hand side
# 1 unit to the right and centred vertically
BBX2A = (1.0, 0.5)

# Font size for plots
FONT_SZ = 36

"""
Functions used for plotting in reports
"""

def sort_cols(df: pd.DataFrame, prefix: str='version', split_tok: str='_'):
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

def dfcol_algoid2ver(df, algoid2ver):
    """ Renames columns in dataframe from algorithm id to version id

    :param df: pandas dataframe
    :param algoid2ver: dictionary of algorithm id -> version id
    :returns: transformed dataframe
    """
    df = df.rename({y: re.sub(r'algorithm_(\d+)', lambda x: f"version_{algoid2ver.get(x.group(1), '0')}", y) for y in df.columns}, axis='columns')
    df = df.groupby(level=0).sum()
    return df

class Plots:
    """ This class is used to generate plots and save them to file
    """


    def __init__(self, plot_dir):
        """ Constructor

        :param plot_dir: filesystem directory used to store plot files
        """
        self.plot_files = {}
        self.plot_dir = plot_dir


    def get_plot_sections(self) -> dict:
        """
        Returns the structure of the plots in the report

        :returns: a dict: value is plot file name, key is report group
        """
        return self.plot_files


    def register_plot(self, group: str, filename: str):
        """ Register a plot
        
        :param group: group in the report that this plot belongs to
        :param file: plot filename
        """
        self.plot_files.setdefault(group, []).append(filename)


    def simple_plot(self, plot_df: pd.DataFrame, plot_file: str, plot_group: str, 
                    title: str, xlabel: str, ylabel: str, show_legend: bool, **plot_kwargs):
        """ Save a simple plot to file

        :param plot_df: Pandas dataframe or series to be plotted
        :param plot_file: plot filename
        :param plot_group: group in the report that this plot belongs to
        :param title: plot title
        :param xlabel: x-axis units label
        :param ylabel: y-axis unit label
        :param show_legend: if True will show a legend, set to False only simple counts are displayed
        :param kwargs: keyword args to pass to the Pandas plot() function
        """
        ax = plot_df.plot(**plot_kwargs, fontsize=FONT_SZ)
        ax.set_title(title, fontsize=FONT_SZ)
        ax.set_xlabel(xlabel, fontsize=FONT_SZ)
        ax.set_ylabel(ylabel, fontsize=FONT_SZ)
        if show_legend:
            ax.legend(fontsize=FONT_SZ, loc='center left', bbox_to_anchor=BBX2A)
            plt.tight_layout(rect=[0, 0, 0.85, 1])
        else:
            plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, plot_file))
        self.register_plot(plot_group, plot_file)


    def plot_borehole_percent(self, nodata_counts: np.array, log1_counts: np.array, all_counts: np.array, provs: np.array):
        """ Save to file percentage of boreholes by provider and data present plot

        :param nodata_counts: nodata counts
        :param log1_counts: log1 counts
        :param all_counts: all counts
        :param provs: provider list
        """
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(1, 1, 1)

        # Calculate log1 counts as a percentage of all counts
        log1_rel = [i / j * 100 for i, j in zip(log1_counts, all_counts)]
        ax.bar(provs, log1_rel, label='HyLogger data')

        # Calculate nodata counts as a percentage of all counts
        nodata_rel = [i / j * 100 for i, j in zip(nodata_counts, all_counts)]
        # Set bottom=log1_rel so it sits on top of log1 bars
        ax.bar(provs, nodata_rel, bottom=log1_rel, label="No HyLogger data")

        # Calculate empty counts as a percentage of all counts
        empty = all_counts-(log1_counts + nodata_counts)
        empty_rel = [i / j * 100 for i, j in zip(empty, all_counts)]
        # Set bottom so it sits on top of log1 & nodata bars
        ax.bar(provs, empty_rel, bottom=[i+j for i,j in zip(log1_rel, nodata_rel)], label="No data")

        # Create bar chart
        plt.ylabel("Percentage of boreholes (%)", fontsize=FONT_SZ)
        plt.title("Percentage of boreholes by provider and data present")
        plt.legend(loc="lower left", bbox_to_anchor=BBX2A)
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.savefig(os.path.join(self.plot_dir, "borehole_percent.png"))
        self.register_plot("Boreholes", "borehole_percent.png")


    def plot_borehole_number(self, all_provs: np.array, all_counts: np.array, title: str, filename: str):
        """ Save to file number of boreholes by provider barchart plot

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
        plt.savefig(os.path.join(self.plot_dir, filename))
        self.register_plot("Boreholes", filename)

    def plot_borehole_geology(self, dfs: dict):
        """ Plot number of boreholes for geology by provider

        :param dfs: dataframe dict, key is log type
        """
       
        dfs_log1_geology = dfs['log1'][dfs['log1']['algorithm'].str.contains(r'^(?:Strat|Form|Lith)', case=False)]
        if not dfs_log1_geology.empty:
            plot_df = dfs_log1_geology.drop_duplicates('nvcl_id').groupby(['provider', 'algorithm']).size().unstack()
            self.split_plots(plot_df, "bar", "Number of boreholes for geology by provider", "Boreholes", "Provider", 1, (30, 15), 36, "log1_geology", "Boreholes", True)


    def plot_borehole_kilometres(self, all_provs: np.array, all_counts: np.array, title: str, filename:str):
        """
        Save to file a number of borehole kilometres by provider plot

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
            if isinstance(h1, float):
                plt.text(r1.get_x() + r1.get_width() / 2., h1, f"{h1:.1f}", ha='center', va='bottom', fontweight='bold')
            else:
                plt.text(r1.get_x() + r1.get_width() / 2., h1, f"{h1}", ha='center', va='bottom', fontweight='bold')
        plt.ylabel("Number of borehole kilometres")
        plt.title(title)
        plt.savefig(os.path.join(self.plot_dir, filename))
        self.register_plot("Boreholes", filename)


    def plot_wordclouds(self, dfs_log2_all: pd.DataFrame):
        """
        Save to file a word clouds plot
    
        :param dfs_log2_all: dataframe to plot
        """
        words = [x.replace(' ','_') for x in list(dfs_log2_all['algorithm'])]
        wordcloud = WordCloud(width=1600, height=800, background_color="white",
                              stopwords = STOPWORDS, max_words=200).generate(' '.join(words))
        plt.figure(figsize=(20, 10))
        plt.imshow(wordcloud, interpolation = 'bilinear')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        plt.savefig(os.path.join(self.plot_dir, 'log2_wordcloud.png'))
        self.register_plot("Word Clouds", 'log2_wordcloud.png')

        words = [x.replace(' ','_') for x in list(dfs_log2_all['algorithm'])]
        wordcloud = WordCloud(width=1600, height=800, background_color="white",
                              stopwords = STOPWORDS, max_words=200).generate(' '.join(words))
        plt.figure(figsize=(20, 10))
        plt.imshow(wordcloud, interpolation = 'bilinear')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        plt.savefig(os.path.join(self.plot_dir, 'log1_wordcloud.png'))
        self.register_plot("Word Clouds", 'log1_wordcloud.png')


    def plot_geophysics(self, dfs_log2_all: pd.DataFrame):
        """
        Save to file a geophysics data by provider plot

        :param dfs_log2_all: dataframe to plot
        """
        plot_group = "Geophysics"
        phys_include = ['magsus', 'mag sus', 'cond']
        df_phys = dfs_log2_all[(dfs_log2_all['algorithm'].str.contains(('|'.join(phys_include)), case=False))]
        if not df_phys.empty:
            plot_df = df_phys.drop_duplicates('nvcl_id')['provider'].value_counts()
            self.simple_plot(plot_df, "geophys_prov.png", plot_group, "Geophysics data by provider",
                      'provider', "Number of boreholes", False, kind='bar', rot=0, figsize=(20, 20))
            plt.close('all')

            # Plot geophysics
            plot_df = df_phys['algorithm'].value_counts()
            self.split_plots(plot_df, "bar", "Geophysics Counts", 'Log 2 Geophysics Algorithms', "Number of data records",
                            5, (20, 20), FONT_SZ, "geophys_count", "Geophysics", False)
            plt.close('all')


    def plot_elements(self, dfs_log2_all):
        """
        Save to file an elements plot

        :param dfs_log2_all: dataframe to plot
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
            plot_df = df_log2_el.drop_duplicates('nvcl_id')['provider'].value_counts()
            self.simple_plot(plot_df, "elems_prov.png", "Elements", "Element data by provider",
                      'provider', "Number of boreholes", False, kind='bar', rot=0, figsize=(20, 20))
    
            # Plot element by number of records
            plot_df = df_log2_el['element'].value_counts()
            if not plot_df.empty:
                self.split_plots(plot_df, "bar", "Elements Graph", 'Element', "Number of data records",
                            20, (40, 20), FONT_SZ, "elems_count", "Elements", False)
    
            # Plot element suffixes sorted by element as separate graphs
            plot_df = df_log2_el.groupby(['element', 'suffix']).size().unstack()
            if not plot_df.empty:
                self.split_plots(plot_df, "bar", "Element suffixes sorted by element", 'Element',
                                 "Number of data records", 1, (40, 30), FONT_SZ, "elems_suffix", "Element Suffixes", True)
    
            # Plot element suffixes for Sulfur
            plot_df = df_log2_el[df_log2_el['element'] == 'S'].groupby(['element', 'suffix']).size().unstack()
            if not plot_df.empty:
                self.simple_plot(plot_df, "elem_S.png", "Element Suffixes", "Element suffixes for Sulfur",
                          'Element', "Number of data records", True, kind='bar', stacked=False, rot=0, figsize=(35, 15))
                plt.close()

            # Plot element suffixes as separate graphs
            plot_df = df_log2_el['suffix'].value_counts()
            if not plot_df.empty:
                self.split_plots(plot_df, "barh", "Element Suffixes", "Number of data records", "Element suffix",
                            20, (90, 90), 100, "elem_suffix_stats", "Element Suffixes", False)
        

    def split_plots(self, plot_df: pd.DataFrame, plot_kind: str , title: str, xlabel: str, ylabel: str, axis_len: int,
                    figsize: (int, int), fontsize: int, file_prefix: str, plot_group: str, show_legend: bool):
        """
        Save to file a series of plots from the one dataframe or series

        :param plot_df: DataFrame or Series to be plotted
        :param plot_kind: either "bar' (vertical bar graph) or 'barh' (horizontal bar graph)
        :param title: plot title
        :param xlabel: x-axis title label
        :param ylabel: y-axis title label
        :param axis_len: split into multiple element graphs 'axis_len' wide ('bar') or high ('barh')
        :param figsize: integer tuple plot size, units are in inches
        :param fontsize: text font size
        :param file_prefix: file name prefix string
        :param plot_group: name of grouping in the report
        :param show_legend: display legend in graph, set to False for graphs with simple counts
        """
        # Split into rows
        if plot_kind == 'bar' or type(plot_df) != pd.DataFrame:
            plot_df_chunks = [plot_df.iloc[i:i + axis_len] for i in range(0, len(plot_df), axis_len)]
        else :
            # Split into columns only if it is a DataFrame
            plot_df_chunks = [plot_df.iloc[:, i:i + axis_len] for i in range(0, plot_df.shape[1], axis_len)]
        MAX_PLOTS = 8
        for idx, df in enumerate(plot_df_chunks):
            if type(plot_df) == pd.DataFrame:
                # Drop columns that have only NaN values
                df = df.dropna(axis=1, how='all')
                # Rename column labels '' to '<Empty>'
                df = df.rename(columns={'': '<Empty>'}) 
            # Rename any '' index labels to '<Empty>'
            df = df.rename(index={'':'<Empty>'})
            ax = df.plot(kind=plot_kind, figsize=figsize, fontsize=fontsize)
            ax.set_title(f"{title} #{idx+1}", fontsize=fontsize)
            ax.set_xlabel(xlabel, fontsize=fontsize)
            ax.set_ylabel(ylabel, fontsize=fontsize)
            # Make sure y-axis is integer, assuming DataFrames always contain simple counts
            if type(plot_df) == pd.DataFrame and plot_df.max().max() < 50.0:
                ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
            if show_legend:
                ax.legend(loc='center left', bbox_to_anchor=BBX2A, fontsize=fontsize)
                plt.tight_layout(rect=[0, 0, 0.85, 1])
            else:
                plt.tight_layout()
            plot_file = f"{file_prefix}_{idx}.png"
            plt.savefig(os.path.join(self.plot_dir, plot_file))
            plt.close()
            self.register_plot(plot_group, plot_file)
            # Exit if done enough plots
            if idx > MAX_PLOTS:
                break


    def plot_spectrum_group(self, dfs, algoid2ver, prefix):
        """
        Save to file spectrum group plots
    
        :param dfs: dataframe to plot
        :param algoid2ver: dictionary of algorithm id -> version id
        """
        # Grp=uTSAS[uTSAS.algorithm.str.startswith('Grp')]
        # [grp_algos, grp_counts] = np.unique(dfs['log1'][dfs['log1'].algorithm.str.startswith('Grp')]['algorithm'], return_counts=True)

        plot_group = "Spectra"
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
            self.simple_plot(plot_df, "grp_utsas.png", plot_group,  "Grp_uTSAS sorted by group name and version",
                      'Group', "Number of data records", True, kind='bar', figsize=(30, 10))

        # TODO: Standardise names before inserted into DataFrame
        plot_df = plot_df.loc[plot_df.index.intersection(['Carbonates', 'CARBONATE'])]
        if not plot_df.empty:
            self.simple_plot(plot_df, "grp_utsas_carbonate.png", plot_group, "Grp_uTSAS sorted by group name and version",
                      'Group', "Number of data records", True, kind='barh', figsize=(20, 10))

        plot_df = Min_uTSAS[sort_cols(Min_uTSAS, prefix)]
        if not plot_df.empty:
            self.simple_plot(plot_df, "min_utsas.png", plot_group, "Min_uTSAS sorted by group name and version",
                      'Mineral', "Number of data records", True, kind='bar', figsize=(30, 10))

        # Plot Min_uTSAS sorted by group name and version
        plot_df = Min_uTSAS[sort_cols(Min_uTSAS, prefix)]
        # TODO: Standardise names before inserted into DataFrame
        plot_df = plot_df.loc[plot_df.index.intersection(["Illitic Muscovite", "Muscovitic Illite", "MuscoviticIllite"])]
        if not plot_df.empty:
            self.simple_plot(plot_df, "min_utsas_misc.png", plot_group, "Min_uTSAS sorted by group name and version",
                      'Group', "Number of data records", True, kind='barh', figsize=(20, 10))
        plt.close('all')


    def plot_algorithms(self, dfs, algoid2ver, prefix='version'):
        """
        Save to file plots for algorithm reports
    
        :param dfs: dataframe to plot
        :param algoid2ver: dictionary of algorithm id -> version id
        """
        plot_group = "Algorithms"
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
            plot_df = dfs_log1_nonstd.drop_duplicates('nvcl_id').groupby(['provider', "Algorithm Prefix"]).size().unstack()
            self.simple_plot(plot_df, "log1_nonstdalgos.png", plot_group, "Number of boreholes for non-standard algorithms by provider",
                 'Provider', "Number of boreholes", True, kind='bar', rot=0, figsize=(50, 40))
        else:
            print("WARNING: There is insufficient data to produce non-standard Log1 algorithm stats")
    
        # Plot number of boreholes by algorithm and provider
        if not df_algo_stats.empty:
            self.split_plots(df_algo_stats, "bar", "Number of boreholes by algorithm and provider",
                             "Provider", "Number of boreholes", 1, (20, 10), FONT_SZ, "log1_algos",
                             plot_group, True)

        # Plot number of data records of standard algorithms by version
        plot_df = df_algoID_stats[sort_cols(df_algoID_stats, prefix)]
        ax = plot_df.plot(kind='bar', stacked=False, figsize=(30, 10), rot=0, fontsize=FONT_SZ)
        ax.set_title("Number of data records of standard algorithms by version", fontsize=FONT_SZ)
        ax.legend(loc="center left", bbox_to_anchor=BBX2A, fontsize=FONT_SZ)
        ax.set_ylabel("Number of data records", fontsize=FONT_SZ)
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.savefig(os.path.join(self.plot_dir, "log1_algoIDs.png"))
        self.register_plot("Algorithms", "log1_algoIDs.png")

        # Plot number of data records of standard algorithms by version and provider
        plot_df = dfs['log1'].groupby(['provider', 'versions']).size().unstack()
        ax = plot_df.plot(kind='bar', stacked=False, figsize=(30, 10), rot=0, fontsize=FONT_SZ)
        ax.set_title("Number of data records of standard algorithms by version and provider", fontsize=FONT_SZ)
        ax.legend(loc='center left', bbox_to_anchor=BBX2A, fontsize=FONT_SZ)
        ax.set_xlabel('Provider', fontsize=FONT_SZ)
        ax.set_ylabel("Number of data records", fontsize=FONT_SZ)
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plot_file = "log1_algoIDs_prov.png"
        plt.savefig(os.path.join(self.plot_dir, plot_file))
        plt.close('all')
        self.register_plot("Algorithms", plot_file)

        # Plot number of data records of algorithmXXX by version and provider
        for alg in df_algo_stats.columns:
            cAlg = dfs['log1'][dfs['log1'].algorithm.str.endswith(alg)]
            # Catch error when no plot data available
            try:
                plot_df = cAlg.drop_duplicates('nvcl_id').groupby(['provider', 'versions']).size().unstack()
                ax = plot_df.plot(kind='bar', stacked=False, figsize=(30, 10), rot=0)
                ax.set_title(f"Number of data records of {alg} by version and provider", fontsize=FONT_SZ)
                ax.legend(loc="center left", bbox_to_anchor=BBX2A, fontsize=FONT_SZ)
                ax.set_xlabel('Provider', fontsize=FONT_SZ)
                ax.set_ylabel("Number of boreholes", fontsize=FONT_SZ)
                plt.tight_layout(rect=[0, 0, 0.85, 1])
                plot_file = f"log1_{alg}-IDs_prov.png"
                plt.savefig(os.path.join(self.plot_dir, plot_file))
                self.register_plot("Algorithms", plot_file)
            except TypeError as ve:
                print(f"WARNING: Cannot plot algorithm {alg}: {ve}")
        plt.close('all')
