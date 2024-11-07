import os
import sys

import pytest
import pandas as pd


# Add in path to source scripts
src_path = os.path.join(os.path.abspath(os.pardir), "src")
sys.path.insert(0, src_path)

from db.tsg_metadata import TSGMeta

from db.readwrite_db import import_db, export_db

@pytest.fixture
def tsg_meta_df():
    """ Provides a TSGMeta dataframe object
    """
    return TSGMeta(os.path.join("data", "metadata.csv")).get_frame()

@pytest.fixture
def tsg_meta_bigger_df():
    """ Provides a TSGMeta dataframe object
    """
    return TSGMeta(os.path.join("data", "metadata_bigger.csv")).get_frame()

@pytest.fixture
def bigger_db_df(tsg_meta_bigger_df):
    """ Provides a bigger dataframe object
    """
    return import_db(os.path.join("data", "bigger.db"), "log1", tsg_meta_bigger_df)

@pytest.fixture
def db_df(tsg_meta_df):
    """ Provides a dataframe object
    """
    return import_db(os.path.join("data", "simple.db"), "log1", tsg_meta_df)

def test_import_db(tsg_meta_df):
    """ Testing import from sqlite db to dataframe
    """
    df = import_db(os.path.join("data", "simple.db"), "log1", tsg_meta_df)
    assert(isinstance(df, pd.DataFrame))
    assert(len(df) == 126)

def test_export_db(db_df, tsg_meta_df):
    """ Can I export, then re-import and the dataframe is still the same?
    """
    export_db("test.db", db_df, "log1", tsg_meta_df)
    db_df_2 = import_db(os.path.join("test.db"), "log1", tsg_meta_df)
    assert(db_df_2.compare(db_df).empty)
    os.remove("test.db")
