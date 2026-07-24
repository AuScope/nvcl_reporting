import os
import sys
from pathlib import Path

import pytest
from psycopg import Connection

from pytest_postgresql import factories


# Add in path to source scripts
src_path = os.path.join(os.path.abspath(os.pardir), "src")
sys.path.insert(0, src_path)

from db.tsg_metadata import TSGMeta

from db.import_db import import_db
from db.export_db import export_db

"""
'conftest.py' is loaded automatically by pytest
Use it to store fixtures that are shared by all tests
"""

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
def bigger_db_df(tsg_meta_bigger_df, postgresql_read):
    """ Provides a bigger dataframe object
    """
    dbparams = { 'user': postgresql_read.info.user, 
                 'host': postgresql_read.info.host, 
                 'port': postgresql_read.info.port,
                 'password': 'password' }
    return import_db(postgresql_read.info.dbname, dbparams, "log1", tsg_meta_bigger_df)

@pytest.fixture
def db_df(tsg_meta_df, postgresql_read):
    """ Provides a dataframe object
    """
    dbparams = { 'user': postgresql_read.info.user, 
                 'host': postgresql_read.info.host, 
                 'port': postgresql_read.info.port,
                 'password': 'password' }
    return import_db(postgresql_read.info.dbname, dbparams, "log1", tsg_meta_df)
