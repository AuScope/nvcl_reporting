import sys
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from urllib.parse import quote_plus
from datetime import date
import json
from datetime import datetime

import pandas as pd

from db.schema import DF_COLUMNS, DATE_FMT

'''
Various routines used for reading and writing to the SQLITE db

Unfortunately there is no automatic conversion between pandas dataframes and database fields with 
more complex types
i.e. pandas won't convert a column of arrays or json for your db even if you use sqlalchemy
'''

def db_col_str() -> str:
    '''
    Makes a comma sep string of column names for converting the database table to a DataFrame
    '''
    # Get dataframe columns
    db_cols = DF_COLUMNS.copy()
    # Remove columns sourced from TSG files
    db_cols.remove('hl_scan_date')
    db_cols.remove('publish_date')
    return ', '.join(db_cols)

def conv_str2dt(dt_str: str) -> date:
    '''
    Converts a date string in the YYYY-MM-DD format to a datetime.date object
    '''
    #assert isinstance(dt_str, str) and dt_str[4] == '/' and dt_str[7] == '/' and dt_str[:3].isnumeric()
    assert type(dt_str) is not pd.Timestamp

    try:
        return datetime.strptime(dt_str, DATE_FMT).date()
    except ValueError:
        dt = datetime.now().date()
        print("dt_str", repr(dt_str), "not a valid string")
        return dt


def conv_str2json(json_str) -> list:
    '''
    Converts from JSON string to Python list object

    :param json_str: JSON string
    :returns: object or [] upon error
    '''
    if json_str == 'nan':
        return []
    try:
        return json.loads(json_str)
    except json.decoder.JSONDecodeError as jde:
        print(jde, "error decoding", repr(json_str))
        sys.exit(1)


def conv_obj2str(arr: []) -> str:
    """ Converts simple list object to JSON-style string """
    try:
        return json.dumps(arr)
    except json.decoder.JSONDecodeError as jde:
        print(f"{jde}: error decoding {arr}")
        sys.exit(9)


def make_engine(db_name: str, db_params: dict) -> Engine:
    user = db_params["user"]
    password = quote_plus(db_params["password"])  # escape special chars
    host = db_params.get("host", "127.0.0.1")
    port = db_params.get("port", 5432)

    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db_name}"
    return create_engine(
        url,
        future=True,
        pool_pre_ping=True,  # good for port-forward / flaky connections
    )
