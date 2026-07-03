import os
import sys
from pathlib import Path

import pytest
import pandas as pd
from psycopg import Connection

from pytest_postgresql import factories


# Add in path to source scripts
src_path = os.path.join(os.path.abspath(os.pardir), "src")
sys.path.insert(0, src_path)

from db.tsg_metadata import TSGMeta

from db.import_db import import_db
from db.export_db import export_db

my_dir = os.path.dirname(os.path.abspath(__file__))

factory_read = factories.postgresql_proc(
    load=[
        Path(my_dir, "data", "meas.sql")
    ]
)
postgresql_read = factories.postgresql("factory_read")

@pytest.fixture
def tsg_meta_df():
    """ Provides a TSGMeta dataframe object
    """
    return TSGMeta(os.path.join(my_dir, "data", "metadata.csv")).get_frame()

@pytest.fixture
def tsg_meta_bigger_df():
    """ Provides a TSGMeta dataframe object
    """
    return TSGMeta(os.path.join(my_dir, "data", "metadata_bigger.csv")).get_frame()

@pytest.fixture
def bigger_db_df(tsg_meta_bigger_df, postgresql_read: Connection):
    """ Provides a bigger dataframe object
    """
    dbparams = { 'user': postgresql_read.info.user, 
                 'host': postgresql_read.info.host, 
                 'port': postgresql_read.info.port,
                 'password': 'password' }
    return import_db(postgresql_read.info.dbname, dbparams, "log1", tsg_meta_bigger_df)

@pytest.fixture
def db_df(tsg_meta_df, postgresql_read: Connection):
    """ Provides a dataframe object
    """
    dbparams = { 'user': postgresql_read.info.user, 
                 'host': postgresql_read.info.host, 
                 'port': postgresql_read.info.port,
                 'password': 'password' }
    return import_db(postgresql_read.info.dbname, dbparams, "log1", tsg_meta_df)



def test_import_db(tsg_meta_df, postgresql_read: Connection):
    """ Testing import from sqlite db to dataframe
    """
    dbparams = { 'user': postgresql_read.info.user, 
                 'host': postgresql_read.info.host, 
                 'port': postgresql_read.info.port,
                 'password': 'password' }
    df = import_db(postgresql_read.info.dbname, dbparams, "log1", tsg_meta_df)
    assert(isinstance(df, pd.DataFrame))
    assert(len(df) == 0)

def test_export_db(db_df, tsg_meta_df, postgresql: Connection):
    """ Can I export, then re-import and the dataframe is still the same?
    """
    pass
    #dbparams = { 'user': postgresql.info.user, 
    #             'host': postgresql.info.host, 
    #             'port': postgresql.info.port,
    #             'password': 'password' }
    #export_db(db_name, db_params, db_df, "log1", tsg_meta_df)
    #db_df_2 = import_db(os.path.join("test.db"), "log1", tsg_meta_df)
    #assert(db_df_2.compare(db_df).empty)
