import sys
import os
import datetime
from pathlib import Path
import yaml

from pyproj.transformer import Transformer

from db.schema import DF_Row
from collections import OrderedDict
from types import SimpleNamespace

def load_and_check_config(config_file: str) -> dict:
    """ Loads config file
    This file contains the directories where the database file is kept, 
    TSG metadata file and the directory where the plot files are kept.
    NB: These files' paths are relative to the config file

    :param config_file: config full path and filename
    :returns: dict of config values
    """
    config_dir = Path(__file__).absolute().parents[1]
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
        # Prepend config directory to make an absolute path
        config[key] = os.path.join(config_dir, config[key])
        # Try to create plot directory
        if key in ('plot_dir') and not os.path.exists(config[key]):
            try:
                os.mkdir(config[key])
            except OSError as oe:
                print(f"Cannot load create directory {config[key]}: {oe}")
                sys.exit(1)
    return config


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

    :param x: longitude in degrees
    :param y: latitude in degrees
    :returns: (x,y) tuple in metres
    """
    transformer = Transformer.from_crs(4326, 7842)
    return transformer.transform(y, x)

def make_row(prov: str, borehole: str, scan_date: datetime.date, modified_date: datetime.date, publish_date: datetime.date):
    """ Returns a DF_Row() instance populated with borehole data abd dates provided

    :param prov: provider string
    :param borehole: dict of borehole metadata
    :param scan_date: HyLogger scan date
    :param modified_date: modified date according to NVCLServices API
    :param publish_date: date TSG file was published at NCI
    :returns: DF_Row() instance 
    """
    easting, northing = to_metres(borehole.x, borehole.y)
    return DF_Row(provider=prov,
        borehole_id=borehole.nvcl_id,
        drill_hole_name=borehole.name,
        publish_date=publish_date,
        hl_scan_date=scan_date,
        easting=easting,
        northing=northing,
        crs="EPSG:7842",
        start_depth=0,
        end_depth=borehole.boreholeLength_m,
        has_vnir=False,
        has_swir=False,
        has_tir=False,
        has_mir=False,
        nvcl_id=borehole.nvcl_id,
        modified_datetime=modified_date,
        log_id='',
        algorithm='',
        log_type='',
        algorithm_id='',
        minerals=[],
        mincnts=[],
        data=[])
