from pyproj.transformer import Transformer
from db.schema import DF_Row
from collections import OrderedDict
from types import SimpleNamespace

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

def make_row(prov, borehole, scan_date, modified_date):
    """ Returns a DF_Row() instance populated with borehole data abd dates provided

    :param prov: provider string
    :param borehole: dict of borehole metadata
    :param scan_date: HyLogger scan date
    :param modified_date: modified date
    :returns: DF_Row() instance 
    """
    easting, northing = to_metres(borehole['x'], borehole['y'])
    return DF_Row(provider=prov,
        borehole_id=borehole['nvcl_id'],
        drill_hole_name=borehole['name'],
        hl_scan_date=scan_date,
        easting=easting,
        northing=northing,
        crs="EPSG:7842",
        start_depth=0,
        end_depth=borehole['boreholeLength_m'],
        has_vnir=False,
        has_swir=False,
        has_tir=False,
        has_mir=False,
        nvcl_id=borehole['nvcl_id'],
        modified_datetime=modified_date,
        log_id='',
        algorithm='',
        log_type='',
        algorithm_id='',
        minerals=[],
        mincnts=[],
        data=[])
