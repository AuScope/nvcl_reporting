# -*- coding: utf-8 -*-

import sys

###
module_paths = ['D:\\Software\\']

for md in module_paths:
    if not(md in sys.path):
        sys.path.append(md)
###

from pathlib import Path
import re
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from itertools import zip_longest
import pickle
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from periodictable import elements
import xml.etree.ElementTree as ET
import xmltodict
import requests
import seaborn as sns
sns.set_context("talk")
try:
    plt.style.use('csiro-basic')
except OSError:
    print("WARNING: Cannot load 'csiro-basic' style")

from nvcl_kit.reader import NVCLReader
from types import SimpleNamespace

# Provider list. Format is (WFS service URL, NVCL service URL, bounding box coords)
prov_list = { 'NSW': ("https://gs.geoscience.nsw.gov.au/geoserver/ows", "https://nvcl.geoscience.nsw.gov.au/NVCLDataServices", None, False, "1.1.0"),
              'NT':  ("http://geology.data.nt.gov.au:80/geoserver/ows", "http://geology.data.nt.gov.au:80/NVCLDataServices", None, True, "2.0.0"),
              'TAS': ("http://www.mrt.tas.gov.au:80/web-services/ows", "http://www.mrt.tas.gov.au/NVCLDataServices/", None, False, "1.1.0"),
              'VIC': ("http://geology.data.vic.gov.au/nvcl/ows", "http://geology.data.vic.gov.au/NVCLDataServices", None, False, "1.1.0"),
              'QLD': ("https://geology.information.qld.gov.au/geoserver/ows", "https://geology.information.qld.gov.au/NVCLDataServices", None, False, "1.1.0"),
              'SA':  ("https://sarigdata.pir.sa.gov.au/geoserver/ows", "https://sarigdata.pir.sa.gov.au/nvcl/NVCLDataServices",None, False, "1.1.0"),
              'WA':  ("http://geossdi.dmp.wa.gov.au/services/ows",  "http://geossdi.dmp.wa.gov.au/NVCLDataServices", None, False, "2.0.0") }

abort_file = Path('./run_NVCL_abort.txt')

ofiles_data = {'log1': "./NVCL_data.pkl",
               'log2': "./NVCL_data_other.pkl",
               'empty': "./NVCL_errors_emptyrecs.pkl",
               'nodata': "./NVCL_errors_nodata.pkl"}
ofiles_stats = {'stats_all': "./NVCL_allstats.pkl",
                'stats_byalgorithms': "./NVCL_algorithm_stats_all.pkl",
                'stats_bystate': "./NVCL_algorithm_stats_bystate.pkl"}

dfs = {}

algo2ver_url = 'https://nvclwebservices.csiro.au/NVCLDataServices/getAlgorithms.html'
r = requests.get(algo2ver_url)
algo2ver_xml = xmltodict.parse(r.content)
algoid2ver_byname = dict()
algoid2ver = dict()
for i in algo2ver_xml['Algorithms']['algorithms']:
    for j in i['outputs']:
        name = j['name']
        ver_dict = dict()
        if isinstance(j['versions'], list):
           for v in j['versions']:
                ver_dict.update({v['algorithmoutputID'] : v['version']})
        else:
            ver_dict.update({j['versions']['algorithmoutputID'] : j['versions']['version']})
        algoid2ver_byname.update({name : ver_dict})
        algoid2ver.update(ver_dict)
algoid2ver['0']='0'

### Internal functions

def create_stats(cdf):
    ca_stats = pd.DataFrame()
    nbores = cdf.loc['nbores']
    if isinstance(nbores,pd.DataFrame):
        nbores = nbores.sum()
    nbores.name = 'nbores'
    ca_stats = ca_stats.append(nbores)
    nmetres = cdf.loc['nmetres']
    if isinstance(nmetres,pd.DataFrame):
        nmetres = nmetres.sum()
    nmetres.name = 'nmetres'
    ca_stats = ca_stats.append(nmetres)
    IDs = np.unique(list(filter(lambda x : re.match("algorithm_\d+",x) ,cdf.index.tolist())))
    for ID in IDs:
        cID = cdf.loc[ID]
        if isinstance(cID,pd.DataFrame):
            cID = cID.sum()
        cID.name = ID
        ca_stats = ca_stats.append(cID)
    return(pd.DataFrame(ca_stats).transpose())

def export_pkl(files2export):
    for outfile in files2export.keys():
        print ('Exporting to '+outfile+' ...')
        if isinstance(files2export[outfile], pd.DataFrame):
            files2export[outfile].to_pickle(outfile)
        else:
            with open(outfile, 'wb') as handle:
                pickle.dump(files2export[outfile], handle)

def import_pkl(infile):
    print ('Importing '+infile+' ...')
    try:
        data = pd.read_pickle(infile)
    except ValueError:
        with open(infile, 'rb') as handle:
            data = pickle.load(handle)
    return(data)

### Primary functions
def read_borehole(prov_list,state,nvcl_id):
    # Optional maximum number of boreholes to fetch, default is no limit
    #MAX_BOREHOLES = 2
    
    columns = ['state','nvcl_id','log_id','algorithm','log_type','algorithmID','minerals','metres','data']

    prov_list[state]    

    param = SimpleNamespace()
    [wfs,nvcl,bbox,local_filt,version] = prov_list[state]

    # URL of the GeoSciML v4.1 BoreHoleView Web Feature Service
    param.WFS_URL = wfs

    # URL of NVCL service
    param.NVCL_URL = nvcl

    # NB: If you set this to true then WFS_VERSION must be 2.0.0
    param.USE_LOCAL_FILTERING = local_filt
    param.WFS_VERSION = version

    # Additional options
    if bbox:
        param.BBOX= bbox
    if 'MAX_BOREHOLES' in locals():
        param.MAX_BOREHOLES = MAX_BOREHOLES

    # Instantiate class and search for boreholes
    reader = NVCLReader(param)

    if not reader.wfs:
        print("ERROR!", wfs, nvcl)

    imagelog_data_list = reader.get_imagelog_data(nvcl_id)
    for ild in imagelog_data_list:
        print (ild.log_name)
    return imagelog_data_list

def read_data(prov_list):
    # Optional maximum number of boreholes to fetch, default is no limit
    #MAX_BOREHOLES = 2
    
    SW_ignore_importedIDs = False
    
    columns = ['state','nvcl_id','log_id','algorithm','log_type','algorithmID','minerals','metres','data']
    
    ids = []
    for df in ofiles_data:
        ofile = ofiles_data[df]
        if Path(ofile).is_file():
            dfs[df] = import_pkl(ofile)
            ids = np.append(ids,dfs[df].nvcl_id.values)
        else:
            dfs[df] = pd.DataFrame(columns=columns)
    
    if abort_file.is_file():
        with open(abort_file, "r") as f:
            remove = f.readlines()
            ids = np.delete(ids, np.argwhere(ids == remove))
    
    print ('Reading NVCL data ...')
    ## read data
    cid = ''
    try:
        for state in prov_list:
            #if not re.match('TAS',state):
            #    continue
    
            print ('\n'+'>'*15+'    '+state+'    '+'<'*15)
            param = SimpleNamespace()
            [wfs,nvcl,bbox,local_filt,version] = prov_list[state]
    
            # URL of the GeoSciML v4.1 BoreHoleView Web Feature Service
            param.WFS_URL = wfs
    
            # URL of NVCL service
            param.NVCL_URL = nvcl

            # NB: If you set this to true then WFS_VERSION must be 2.0.0
            param.USE_LOCAL_FILTERING = local_filt
            param.WFS_VERSION = version

            # Additional options
            if bbox:
                param.BBOX= bbox
            if 'MAX_BOREHOLES' in locals():
                param.MAX_BOREHOLES = MAX_BOREHOLES
    
            # Instantiate class and search for boreholes
            reader = NVCLReader(param)

            if not reader.wfs:
                print("ERROR!", wfs, nvcl)
    
            bh_list = reader.get_boreholes_list()
            nvcl_id_list = reader.get_nvcl_id_list()
    
            # Get NVCL log id for first borehole in list
            if not nvcl_id_list:
                print("!!!! No NVCL ids for", nvcl)
                continue
    
            for iID in range(len(nvcl_id_list)):
                nvcl_id = nvcl_id_list[iID]
                print ('-'*50)
                print (nvcl_id,' - ',state,'('+str(iID+1),'of',str(len(nvcl_id_list))+')')
                print ('-'*10)
                cid = nvcl_id
                if (SW_ignore_importedIDs == True and nvcl_id in ids):
                        print ('Already imported, next...')
                else:
                    imagelog_data_list = reader.get_imagelog_data(nvcl_id)
                    if not imagelog_data_list:
                        print ('No NVCL data!')
                        data = [state, nvcl_id, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
                        dfs['nodata'] = dfs['nodata'].append(pd.Series(data, index=dfs['nodata'].columns), ignore_index=True)
                    for ild in imagelog_data_list:
                        print (ild.log_name)
                        if ((ild.log_id in dfs['log1'].log_id.values) or (ild.log_id in dfs['empty'].log_id.values)):
                            print ('Already imported, next...')
                            continue
                        HEIGHT_RESOLUTION = 1.0
                        ANALYSIS_CLASS = ''
                        #ANALYSIS_CLASS = 'Min1 uTSAS'
                        #if ild.log_type == LOG_TYPE and ild.algorithm == ANALYSIS_CLASS:
                        minerals = []
                        nmetres = []
                        if ild.log_type == '1':
                            bh_data = reader.get_borehole_data(ild.log_id, HEIGHT_RESOLUTION, ANALYSIS_CLASS)
                            if bh_data:
                                [minerals, nmetres] = list(np.unique([getattr(bh_data[i], 'classText',"Unknown") for i in bh_data.keys()], return_counts=True))
                            data = [state, nvcl_id, ild.log_id, ild.log_name, ild.log_type, ild.algorithmout_id, minerals, nmetres, bh_data]
                            #print(data)
                        else:
                            data = [state, nvcl_id, ild.log_id, ild.log_name, ild.log_type, ild.algorithmout_id, np.nan, np.nan, np.nan]
        
                        if len(minerals) > 0:
                            dfs['log'+ild.log_type] = dfs['log'+ild.log_type].append(pd.Series(data, index=dfs['log'+ild.log_type].columns), ignore_index=True)
                        else:
                            dfs['empty'] = dfs['empty'].append(pd.Series(data, index=dfs['empty'].columns), ignore_index=True)
                    np.append(ids,nvcl_id)
    except KeyboardInterrupt:
        with open(abort_file, "w") as f:
            f.write(cid)
        for df in ofiles_data:
            ofile = ofiles_data[df]
            export_pkl({ofile: dfs[df]})
        sys.exit()
    
    for df in ofiles_data:
        ofile = ofiles_data[df]
        export_pkl({ofile: dfs[df]})

    return dfs

def calc_stats(prov_list, dfs):
    df_allstats = pd.DataFrame()
    ## munge data
    print ('Calculating initial statistics ...')
    for state in prov_list:
        cdf = dfs['log1'][dfs['log1']['state']==state]
        algorithms = np.unique(cdf['algorithm'])
    
        for algorithm in algorithms:
            acdf = cdf[cdf.algorithm == algorithm]
            df_nbores=pd.DataFrame.from_records(zip_longest(*acdf.minerals.values))
            df_nmetres=pd.DataFrame.from_records(zip_longest(*acdf.metres.values))
            df_algorithmID=acdf.algorithmID
            df_temp=df_nbores.apply(pd.Series.value_counts)
            df_nborescount=df_temp.sum(axis=1)
    
            df_cstats = pd.DataFrame(columns=['nbores','nmetres'])
            df_cstats['nbores'] = df_nborescount
     
            if len(df_algorithmID) == 1:
                IDs = df_algorithmID.to_list()*len(df_cstats)
            else:
                IDs = df_algorithmID.to_list()
            algos = []
            for iC, col in df_nbores.iteritems():
                #print (IDs[iC])
                algo = 'algorithm_'+IDs[iC]
                algos.append(algo)
                if not algo in df_cstats:
                    df_cstats.insert(0,algo,0*len(df_cstats))
                df_cstats[algo].loc[col.dropna().tolist()] += 1
            df_cstats[np.unique(algos)] = df_cstats[np.unique(algos)].replace({0:np.nan})
    
            for row, value in df_nborescount.iteritems():
                df_cstats.loc[row,'nmetres']=np.nansum(df_nbores.isin([row])*df_nmetres.values)
    
            df_cstats = df_cstats.transpose()
            df_cstats['state']=state
            df_cstats['algorithm']=algorithm
            df_cstats['stat']=df_cstats.index
    
            #df_allstats = df_allstats.append(df_cstats.set_index(['state', 'algorithm', 'stat'], inplace=False), sort=True)
            df_allstats = df_allstats.append(df_cstats, ignore_index=True, sort=False)
    
    df_allstats=df_allstats.set_index(['state', 'algorithm', 'stat'])
    export_pkl({ofiles_stats['stats_all']: df_allstats})
    
    algorithm_stats_all = dict()
    algorithms = np.unique(dfs['log1']['algorithm'])
    print ('Calculating algorithm based statistics ...')
    for algorithm in algorithms:
        cdf = df_allstats.xs(algorithm,level='algorithm').dropna(axis=1,how='all').droplevel(0)
        algorithm_stats_all[algorithm] = create_stats(cdf)

    export_pkl({ofiles_stats['stats_byalgorithms']: algorithm_stats_all})
    
    algorithm_stats_bystate = dict()
    print ('Calculating state based statistics ...')
    for algorithm in algorithms:
        algorithm_stats_bystate[algorithm] = dict()
        sdf = df_allstats.xs(algorithm,level='algorithm')
        states = np.unique(dfs['log1'][dfs['log1']['algorithm']==algorithm]['state'])
        for state in states:
            cdf = sdf.xs(state,level='state').dropna(axis=1,how='all')
            algorithm_stats_bystate[algorithm][state] = create_stats(cdf)
    
    export_pkl({ofiles_stats['stats_bystate']: algorithm_stats_bystate})
    if abort_file.is_file():
        abort_file.unlink()
    
    dfs['stats_all'] = df_allstats
    dfs['stats_byalgorithms'] = algorithm_stats_all
    dfs['stats_bystate'] = algorithm_stats_bystate

    return (dfs)

def plot_results(dfs):
    df_all = pd.concat([dfs['log1'], dfs['log2'], dfs['empty'], dfs['nodata']])
    dfs_log2_all = pd.concat([dfs['log2'],dfs['empty'][dfs['empty']['log_type']=='2']])
    [all_states, all_counts] = np.unique(df_all.drop_duplicates(subset='nvcl_id')['state'], return_counts=True)
    [log1_states, log1_counts] = np.unique(dfs['log1'].drop_duplicates(subset='nvcl_id')['state'],return_counts=True)
    [log2_states, log2_counts] = np.unique(dfs['log2'].drop_duplicates(subset='nvcl_id')['state'],return_counts=True)
    [nodata_states, nodata_counts] = np.unique(dfs['nodata'].drop_duplicates(subset='nvcl_id')['state'],return_counts=True)
    df_empty_log1 = dfs['empty'][dfs['empty']['log_type']=='1']
    [empty_states, empty_counts] = np.unique(df_empty_log1.drop_duplicates(subset='nvcl_id')['state'],return_counts=True)

    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(1, 1, 1)
    log1_rel = [i / j * 100 for  i,j in zip(log1_counts, all_counts)]
    p1 = ax.bar(log1_states, log1_rel, label='HyLogger data')
    nodata_rel = [i / j * 100 for  i,j in zip(nodata_counts, all_counts)]
    p2 = ax.bar(nodata_states, nodata_rel, bottom=log1_rel, label='No HyLogger data')
    empty = all_counts-(log1_counts+nodata_counts)
    empty_rel = [i / j * 100 for  i,j in zip(empty, all_counts)]
    p3 = ax.bar(empty_states, empty_rel, bottom=[i+j for i,j in zip(log1_rel, nodata_rel)], label='No data')
    plt.ylabel('Percentage of boreholes (%)')
    plt.title('Percentage of boreholes by state and data present')
    plt.legend(loc='lower left')
    plt.savefig('borehole_percent.png')

    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(1, 1, 1)
    ax1 = ax.bar(all_states, all_counts)
    for r1 in ax1:
        h1 = r1.get_height()
        plt.text(r1.get_x() + r1.get_width() / 2., h1, "%d" % h1, ha="center", va="bottom", fontweight="bold")
    plt.ylabel('Number of boreholes')
    plt.title('Number of boreholes by state')
    plt.savefig('borehole_number.png')

    """
    words = [x.replace(' ','_') for x in list(dfs_log2_all['algorithm'])]
    wordcloud = WordCloud(width=1600, height=800, background_color="white", stopwords = STOPWORDS, max_words=200).generate(' '.join(words))
    plt.figure(figsize=(20,10))
    plt.imshow(wordcloud, interpolation = 'bilinear')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    plt.savefig('log2_wordcloud.png')

    words = [x.replace(' ','_') for x in list(dfs_log2_all['algorithm'])]
    wordcloud = WordCloud(width=1600, height=800, background_color="white", stopwords = STOPWORDS, max_words=200).generate(' '.join(words))
    plt.figure(figsize=(20,10))
    plt.imshow(wordcloud, interpolation = 'bilinear')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    plt.savefig('log1_wordcloud.png')
    """

    algos = np.unique(dfs['log1'][dfs['log1'].algorithm.str.contains('^Min|Grp')]['algorithm'])
    suffixes = np.unique([x.split()[1] for x in algos])
    dfs['log1']['versions']=dfs['log1'].apply(lambda row: algoid2ver[row['algorithmID']], axis=1)

    df_algo_stats = pd.DataFrame()
    df_algoID_stats = pd.DataFrame()
    for suffix in suffixes:
        [states,count] = np.unique(dfs['log1'][dfs['log1'].algorithm.str.endswith(suffix)].drop_duplicates('nvcl_id')['state'], return_counts=True)
        df_algo_stats=pd.concat([df_algo_stats,pd.DataFrame({suffix: count}, index=states)], axis=1, sort=False)
        [IDs,count] = np.unique(dfs['log1'][dfs['log1'].algorithm.str.endswith(suffix)]['versions'], return_counts=True)
        #IDs = ['algorithm_'+x for x in IDs]
        vers = ['version_'+x for x in IDs]
        df_algoID_stats=pd.concat([df_algoID_stats,pd.DataFrame([np.array(count)],columns=vers,index=[suffix])], sort=False)

    dfs_log1_geology = dfs['log1'][dfs['log1']['algorithm'].str.contains(('^(Strat|Form|Lith)'), case=False)]
    ax = dfs_log1_geology.drop_duplicates('nvcl_id').groupby(['state', 'algorithm']).size().unstack().plot(kind='bar', rot=0, figsize=(20,10), title='Number of boreholes for geology by state')
    ax.set(xlabel='State', ylabel='Number of boreholes')
    plt.tight_layout()
    plt.savefig('log1_geology.png')

    dfs_log1_nonstd = dfs['log1'][~(dfs['log1']['algorithm'].str.contains(('^(Grp|Min|Sample|Lith|HoleID|Strat|Form)'), case=False))]
    dfs_log1_nonstd['Algorithm Prefix'] = dfs_log1_nonstd['algorithm'].replace({'(grp_|min_)' : ''}, regex=True).replace({'_*\d+$' : ''}, regex=True)
    ax = dfs_log1_nonstd.drop_duplicates('nvcl_id').groupby(['state', 'Algorithm Prefix']).size().unstack().plot(kind='bar', rot=0, figsize=(20,10), title='Number of boreholes for non-standard algorithms by state')
    ax.set(xlabel='State', ylabel='Number of boreholes')
    plt.tight_layout()
    plt.savefig('log1_nonstdalgos.png')

    ax = df_algo_stats.plot(kind='bar', stacked=False, figsize=(20,10), rot=0, title='Number of boreholes by algorithm and state')
    ax.set(ylabel='Number of boreholes')
    #for p in ax.patches:
    #    ax.annotate(str(int(p.get_height())), (p.get_x()+p.get_width()/2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points', size=4, fontweight="bold")
    plt.tight_layout()
    plt.savefig('log1_algos.png')

    ax = df_algoID_stats[sort_cols(df_algoID_stats)].plot(kind='bar', stacked=False, figsize=(30,10), rot=0, title='Number of data records of standard algorithms by version')
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    #ax.grid(True, which='major', linestyle='-')
    #ax.grid(True, which='minor', linestyle='--')
    ax.set(ylabel='Number of data records')
    plt.tight_layout()
    plt.savefig('log1_algoIDs.png')

    ax = dfs['log1'].groupby(['state','versions']).size().unstack().plot(kind='bar', stacked=False, figsize=(30,10), rot=0, title='Number of data records of standard algorithms by version and state')
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    ax.set(xlabel='State', ylabel='Number of data records')
    plt.tight_layout()
    plt.savefig('log1_algoIDs_state.png')

    plt.close('all')
    for alg in df_algo_stats.columns:
        cAlg=dfs['log1'][dfs['log1'].algorithm.str.endswith(alg)]
        ax = cAlg.drop_duplicates('nvcl_id').groupby(['state','versions']).size().unstack().plot(kind='bar', stacked=False, figsize=(30,10), rot=0, title='Number of data records of '+alg+' by version and state')
        ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        ax.set(xlabel='State', ylabel='Number of boreholes')
        plt.tight_layout()
        plt.savefig('log1_'+alg+'-IDs_state.png')

    #Grp=uTSAS[uTSAS.algorithm.str.startswith('Grp')]
    #[grp_algos, grp_counts] = np.unique(dfs['log1'][dfs['log1'].algorithm.str.startswith('Grp')]['algorithm'], return_counts=True)

    Grp_uTSAS = pd.DataFrame()
    for key, value in dfs['stats_byalgorithms'].items():
        if key.startswith('Grp') and key.endswith('uTSAS'):
            Grp_uTSAS = pd.concat([Grp_uTSAS, value]).fillna(0).groupby(level=0).sum()
    Grp_uTSAS=dfcol_algoid2ver(Grp_uTSAS)
    Min_uTSAS = pd.DataFrame()
    for key, value in dfs['stats_byalgorithms'].items():
        if key.startswith('Min') and key.endswith('uTSAS'):
            Min_uTSAS = pd.concat([Min_uTSAS, value]).fillna(0).groupby(level=0).sum()
    Min_uTSAS=dfcol_algoid2ver(Min_uTSAS)

    plt.close('all')
    ax = Grp_uTSAS[sort_cols(Grp_uTSAS)].plot(kind='bar', figsize=(30,10), title='Grp_uTSAS sorted by group name and version')
    ax.set(xlabel='Group', ylabel='Number of data records')
    plt.tight_layout()
    plt.savefig('grp_utsas.png')
    ax = Grp_uTSAS[sort_cols(Grp_uTSAS)].loc[['Carbonates','CARBONATE']].plot(kind='barh', figsize=(20,10), title='Grp_uTSAS sorted by group name and version')
    ax.set(ylabel='Group', xlabel='Number of data records')
    plt.tight_layout()
    plt.savefig('grp_utsas_carbonate.png')
    ax = Min_uTSAS[sort_cols].plot(kind='bar', figsize=(30,10), title='Min_uTSAS sorted by group name and version')
    ax.set(xlabel='Mineral', ylabel='Number of data records')
    plt.tight_layout()
    plt.savefig('min_utsas.png')
    ax = Min_uTSAS[sort_cols(Grp_uTSAS)].loc[['Illitic Muscovite','Muscovitic Illite','MuscoviticIllite']].plot(kind='barh', figsize=(20,10), title='Min_uTSAS sorted by group name and version')
    ax.set(ylabel='Group', xlabel='Number of data records')
    plt.tight_layout()
    plt.savefig('min_utsas_musc.png')

    #pd.DataFrame({'algo':grp_algos, 'counts':grp_counts}).plot.bar(x='algo',y='counts', rot=20)
    #[min_algos, min_counts] = np.unique(dfs['log1'][dfs['log1'].algorithm.str.startswith('Min')]['algorithm'], return_counts=True)
    #pd.DataFrame({'algo':min_algos, 'counts':min_counts}).plot.bar(x='algo',y='counts', rot=20)
    #index = [x.replace('Grp','') for x in grp_algos]
    #df = pd.DataFrame({'Grp': grp_counts, 'Min': min_counts}, index=index).plot.bar(rot=20)

    el_exclude = np.concatenate((suffixes,['white mica','featex','mWt','Chlorite Epidote Index','albedo','Core colour from imagery','ISM','Kaolinite Crystallinity','Mica','smooth','TIR','magsus','mag sus','cond','reading','pfit','color']))
    df_excluded = dfs_log2_all[~(dfs_log2_all['algorithm'].str.contains(('|'.join(el_exclude)), case=False))]
    df_log2_el = dfs_log2_all[(dfs_log2_all['algorithm'].str.contains(('|'.join(['ppm','pct','%','per','arsen','Sillimanite','PbZn'])), case=False))]
    df_log2_el = df_log2_el.append(dfs_log2_all[(dfs_log2_all['algorithm'].str.match('^('+('|'.join([str(x) for x in list(elements)]))+')$', case=False))])
    df_log2_el = df_log2_el.append(dfs_log2_all[(dfs_log2_all['algorithm'].str.match('^('+('|'.join([str(x) for x in list(elements)]))+')\s+.*$', case=False))])
    df_diff = df_excluded[~df_excluded.apply(tuple,1).isin(df_log2_el.apply(tuple,1))]
    df_log2_el['element'] = df_log2_el['algorithm'].replace({'(?i)ppm|(?i)pct|(?i)per' : r''}, regex=True).replace({'([A-Za-z0-9]+)(_| )*(.*)' : r'\1'}, regex=True).replace({'(\w+)(0|1|SFA)' : r'\1'}, regex=True)
    df_log2_el['suffix'] = [a.replace(b, '') for a, b in zip(df_log2_el['algorithm'], df_log2_el['element'])]
    df_log2_el['suffix'] = df_log2_el['suffix'].replace({'^(_.*)' : r' \1'}, regex=True)
    df_log2_el['element'] = df_log2_el['element'].replace({'(?i)Arsen$' : 'Arsenic'}, regex=True).apply(lambda x: (x[0].upper()+x[1].lower()+x[2:]) if len(x)>2 else x[0].upper()+x[1].lower() if len(x)>1 else x[0].upper())

    plt.close('all')
    ax = df_log2_el.drop_duplicates('nvcl_id')['state'].value_counts().plot(kind='bar', rot=0, figsize=(10,10), title='Element data by state')
    ax.set(xlabel='state', ylabel='Number of boreholes')
    plt.tight_layout()
    plt.savefig('elems_state.png')
    ax = df_log2_el['element'].value_counts().plot(kind='bar', figsize=(20,10), title='Elements')
    ax.set(xlabel='Element', ylabel='Number of data records')
    plt.tight_layout()
    plt.savefig('elems_count.png')
    ax = df_log2_el.groupby(['element', 'suffix']).size().unstack().plot(kind='bar', stacked=False, figsize=(30,10), title='Element suffixes sorted by element')
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    ax.set(xlabel='Element', ylabel='Number of data records')
    plt.tight_layout()
    plt.savefig('elems_suffix.png')
    ax = df_log2_el[df_log2_el['element']=='S'].groupby(['element', 'suffix']).size().unstack().plot(kind='bar', stacked=False, rot=0, figsize=(10,10), title='Element suffixes for Sulfur')
    ax.set(xlabel='Element', ylabel='Number of data records')
    plt.tight_layout()
    plt.savefig('elem_S.png')
    plt.close('all')
    ax = df_log2_el['suffix'].value_counts().plot(kind='barh', figsize=(20,10), title='Element suffixes')
    ax.set(ylabel='Element suffix', xlabel='Number of data records')
    plt.tight_layout()
    plt.savefig('elem_suffix_stats.png')

    plt.close('all')
    phys_include = ['magsus','mag sus','cond']
    df_phys = dfs_log2_all[(dfs_log2_all['algorithm'].str.contains(('|'.join(phys_include)), case=False))]
    ax = df_phys.drop_duplicates('nvcl_id')['state'].value_counts().plot(kind='bar', rot=0, figsize=(10,10), title='Geophysics data by state')
    ax.set(xlabel='state', ylabel='Number of boreholes')
    plt.tight_layout()
    plt.savefig('geophys_state.png')
    plt.close('all')
    ax = df_phys['algorithm'].value_counts().plot(kind='bar', figsize=(10,10), rot=90, title='Geophysics')
    ax.set(xlabel='data', ylabel='Number of data records')
    plt.tight_layout()
    plt.savefig('geophys_count.png')
    plt.close('all')

def dfcol_algoid2ver(df):
    df=df.rename({y: re.sub(r'algorithm_(\d+)', lambda x: 'version_' + algoid2ver[x.group(1)], y) for y in df.columns}, axis='columns')
    df=df.fillna(0).groupby(level=0).sum()
    return df

def sort_cols(df,prefix='version',split='_'):
    cols = sorted(df.columns)
    anums = list()
    for c in cols:
        if (re.match('^'+prefix,c)):
            anums.append(c.split(split)[1])
    return ([prefix+split+str(x) for x in sorted([int(x) for x in anums])])

if __name__ == "__main__":
    sw_read_data = True
    sw_calc_stats = True
    sw_plot = True
    sw_load_data = True

    #data = read_borehole(prov_list,'NT','8601229_MCDD0005')

    if sw_read_data:
        dfs = read_data(prov_list)
    else:
        if sw_load_data == True:
            for df in ofiles_data:
                dfs[df] = import_pkl(ofiles_data[df])

    if sw_calc_stats:
        dfs = calc_stats(prov_list, dfs)
    else:
        if sw_load_data == True:
            for df in ofiles_stats:
                dfs[df] = import_pkl(ofiles_stats[df])

    if sw_plot:
        plot_results(dfs)
