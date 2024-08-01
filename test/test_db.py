#def export_db(db_file: str, df: pd.DataFrame, report_category: str, known_id_df: pd.DataFrame):
import os
import sys

# Add in path to source scripts
src_path = os.path.join(os.path.abspath(os.pardir), 'src')
sys.path.insert(0, src_path)

import pytest
import pandas as pd

from db.readwrite_db import import_db, export_db

@pytest.fixture
def db_df():
    return import_db(os.path.join("data","simple.db"), "log1")

def test_import_db():
    """ Testing import from sqlite db to dataframe
    """
    df = import_db(os.path.join("data","simple.db"), "log1")
    assert(isinstance(df, pd.DataFrame))
    assert(len(df) == 126)

def test_export_db(db_df):
    """ Can I export, then re-import and the dataframe is still the same?
    """
    export_db("test.db", db_df, "log1", pd.DataFrame())
    db_df_2 = import_db(os.path.join("test.db"), "log1")
    assert(db_df_2.compare(db_df).empty)



