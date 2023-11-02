'''
This is the schema for the database
'''

import json
import sys
import math
from collections import OrderedDict
from typing import Iterable
from dataclasses import dataclass
from datetime import datetime
from types import SimpleNamespace

from peewee import Field, Model, TextField, DateField, CompositeKey, BooleanField, DoubleField, FloatField

# Database date format
DATE_FMT = '%Y-%m-%d'

# Dataframe columns
DF_COLUMNS = ['provider', 'borehole_id', 'drill_hole_name', 'hl_scan_date', 'easting', 'northing', 'crs', 'start_depth', 'end_depth', 'has_vnir', 'has_swir', 'has_tir', 'has_mir', 'nvcl_id', 'modified_datetime', 'log_id', 'algorithm', 'log_type', 'algorithm_id', 'minerals', 'mincnts', 'data']


class ScalarsField(Field):
    field_type = 'Scalars'

    def db_value(self, data: any) -> str:
        '''
        Convert mineral types at each depth to JSON formatted string
        i.e. ((depth, {key: val ...}, ...) ... ) or NaN

        :param data: mineral types at each depth
        :returns: JSON formatted string
        '''
        data_list = []
        if isinstance(data, OrderedDict):
            for depth, obj in data.items():
                # vars() converts Namespace -> dict
                if isinstance(obj, SimpleNamespace):
                    data_list.append([depth, vars(obj)])
                elif isinstance(obj, list) and len(obj) == 0:
                    continue
                else:
                    print(repr(obj), type(obj))
                    print("ERROR Unknown obj type in 'data' var")
                    sys.exit(1)
        elif data != {} and data != [] and not isinstance(data, list) and not (isinstance(data, float) and math.isnan(data)):
            print(repr(data), type(data))
            print("ERROR Unknown type in 'data' var")
            sys.exit(1)

        return json.dumps(data_list)

    def python_value(self, value: str) -> any:
        ''' 
        Converts from JSON string to Python object

        :param value: JSON string
        :returns: object or [] upon error
        '''
        #try:
        return json.loads(value) # Convert JSON string to python obj
        #except json.decoder.JSONDecodeError:
        #    return []



class JSONField(Field):
    field_type = 'JSON'

    def db_value(self, values: any) -> str:
        '''
        Converts lists of values to JSON formatted string

        :param values: list of valuex
        :returns: JSON formatted string
        '''
        # Sometimes 'values' is not an iterable numpy array
        if isinstance(values, Iterable):
            value_out = json.dumps(list(values))
        elif isinstance(values, float) and math.isnan(values):
            value_out = '[]'
        else:
            value_out = json.dumps([values])
        return value_out

    def python_value(self, value: str) -> any:
        ''' 
        Converts from JSON string to Python object

        :param value: JSON string
        :returns: object or [] upon error
        '''
        #try:
        return json.loads(value) # Convert JSON string to python obj
        #except json.decoder.JSONDecodeError:
        #    return []


class Meas(Model):
    # NB: 'report_category' field is not stored when converted to a DataFrame
    report_category = TextField() # Can be any one of 'log1', 'log2', 'log6', 'empty' and 'nodata'
    provider = TextField()
    borehole_id = TextField()
    drill_hole_name = TextField()
    hl_scan_date = DateField(formats=[DATE_FMT]) # TODO: Have to get this from TSG files
    easting = DoubleField()
    northing = DoubleField()
    crs = TextField() # EPSG:7842
    start_depth = FloatField()
    end_depth = FloatField()
    has_vnir = BooleanField()
    has_swir = BooleanField()
    has_tir = BooleanField()
    has_mir = BooleanField()
    nvcl_id = TextField()
    modified_datetime = DateField(formats=[DATE_FMT]) # Only some providers supply it, else retrieval date is used
    log_id = TextField()
    algorithm = TextField()
    log_type = TextField()
    algorithm_id = TextField()
    minerals = JSONField() # Unique minerals
    mincnts = JSONField()  # Counts of unique minerals as an array
    data = ScalarsField()     # Mineral scalars data 

    class Meta:
        primary_key = CompositeKey('report_category', 'provider', 'nvcl_id', 'log_id', 'algorithm', 'log_type', 'algorithm_id')
        database = None


# 'dataclass' annotation automatically provides __init__ & __repr__
@dataclass
class DF_Row:
    """
    Class for DataFrame rows
    """
    provider: str
    borehole_id: str
    drill_hole_name: str
    hl_scan_date: datetime.date # Hylogger scan date
    easting: float
    northing: float
    crs: str # CRS for northing & easting
    start_depth: float
    end_depth: float
    has_vnir: bool # has Very Near IR data
    has_swir: bool # has Short Wave IR data
    has_tir: bool # has Thermal IR data
    has_mir: bool # has Mid IR data
    nvcl_id: str
    modified_datetime: datetime # Only some providers supply it, else retrieval date is used
    log_id: str
    algorithm: str
    log_type: str
    algorithm_id: str
    minerals: list # Unique minerals
    mincnts: dict   # Counts of unique minerals as an array
    data: SimpleNamespace # Mineral scalars data

    def as_list(self):
        """
        Return the attribute values as a list
        """
        # TODO: Systematise, too many lists
        ATTR_LIST = ["provider",
    "borehole_id",
    "drill_hole_name",
    "hl_scan_date",
    "easting",
    "northing",
    "crs",
    "start_depth",
    "end_depth",
    "has_vnir",
    "has_swir",
    "has_tir",
    "has_mir",
    "nvcl_id",
    "modified_datetime",
    "log_id",
    "algorithm",
    "log_type",
    "algorithm_id",
    "minerals",
    "mincnts",
    "data"]

        return [getattr(self,attr) for attr in ATTR_LIST]

