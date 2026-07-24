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


def test_import_db(tsg_meta_df, postgresql_read):
    """ Testing import from sqlite db to dataframe
    """
    dbparams = { 'user': postgresql_read.info.user, 
                 'host': postgresql_read.info.host, 
                 'port': postgresql_read.info.port,
                 'password': 'password' }
    df = import_db(postgresql_read.info.dbname, dbparams, "log1", tsg_meta_df)
    assert(isinstance(df, pd.DataFrame))
    assert(len(df) == 0)

def test_export_db(db_df, tsg_meta_df, postgresql_read):
    """ Can I export, then re-import and the dataframe is still the same?
    """
    pass
    #dbparams = { 'user': postgresql_read.info.user, 
    #             'host': postgresql_read.info.host, 
    #             'port': postgresql_read.info.port,
    #             'password': 'password' }
    #export_db(db_name, db_params, db_df, "log1", tsg_meta_df)
    #db_df_2 = import_db(os.path.join("test.db"), "log1", tsg_meta_df)
    #assert(db_df_2.compare(db_df).empty)
