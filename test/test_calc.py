import os
import sys
import datetime
import math

import pytest

# Add in path to source scripts
src_path = os.path.join(os.path.abspath(os.pardir), 'src')
sys.path.insert(0, src_path)

from calculations import calc_bh_depths, calc_fyq, get_fy_date_ranges
# Don't delete 'tsg_meta_df' 'tsg_meta_bigger_df', needed for fixtures to be injected
from test_db import db_df, bigger_db_df, tsg_meta_df, tsg_meta_bigger_df
from db.readwrite_db import import_db


def test_calc_bh_depths_cnts(db_df):
    """ Testing depth calculation
    ID:24582 - 0.5m  6.5m = 5 + 1 = 0.006
    ID:24598 - 1.5m  7.5m = 6 + 1 = 0.007
    ID:24725 - 0.5m  64.5m = 64 + 1 = 0.065
    ID:24998 - 0.5m  494.5m = 494 + 1 = 0.495
    ID:24999 - 0.5m  401.5m = 401 + 1 = 0.402
                                   SUM: 0.975
    """
    df_dict = { 'log1': db_df }
    depth = calc_bh_depths(df_dict, 'TAS', 'publish_date')
    assert math.isclose(depth, 0.975)
    cnts, depth = calc_bh_depths(df_dict, 'TAS', 'publish_date', return_cnts=True)
    assert math.isclose(depth, 0.975)
    assert cnts == 5


def test_calc_bh_depths_no_cnts(db_df):
    df_dict = { 'log1': db_df }
    depth = calc_bh_depths(df_dict, 'QLD', 'publish_date')
    assert math.isclose(depth, 0.0)
    cnts, depth = calc_bh_depths(df_dict, 'QLD', 'publish_date', return_cnts=True)
    assert math.isclose(depth, 0.0)
    assert cnts == 0

def test_calc_bh_depths_date_ranges(db_df):
    """ Testing date ranges
    24582|2022-06-23 0.006
    24598|2022-06-25 0.007
    24725|2023-06-27 0.065
    24998|2022-06-29 0.495
    24999|2022-07-01 0.402
    """
    df_dict = { 'log1': db_df }
    depth = calc_bh_depths(df_dict, 'TAS', 'publish_date',
                           start_date=datetime.date(2022, 6, 25))
    assert math.isclose(depth, 0.962)
    cnts, depth = calc_bh_depths(df_dict, 'TAS', 'publish_date',
                                 start_date=datetime.date(2022, 6, 25), return_cnts=True)
    assert math.isclose(depth, 0.962)
    assert cnts == 3

    depth = calc_bh_depths(df_dict, 'TAS', 'publish_date',
                           start_date=datetime.date(2022, 6, 24),
                           end_date=datetime.date(2022, 6, 28))
    assert math.isclose(depth, 0.072)
    cnts, depth = calc_bh_depths(df_dict, 'TAS', 'publish_date',
                                 start_date=datetime.date(2022, 6, 24),
                                 end_date=datetime.date(2022, 6, 28), return_cnts=True)
    assert math.isclose(depth, 0.072)
    assert cnts == 2

    depth = calc_bh_depths(df_dict, 'TAS', 'publish_date',
                           end_date=datetime.date(2022, 6, 27))
    assert math.isclose(depth, 0.013)
    cnts, depth = calc_bh_depths(df_dict, 'TAS', 'publish_date',
                                 end_date=datetime.date(2022, 6, 27), return_cnts=True)
    assert math.isclose(depth, 0.013)
    assert cnts == 2


def test_get_fy_date_ranges():
     # Jan 12 2024 - first quarter of calendar year
     y_start, y_end, q_start, q_end = get_fy_date_ranges(datetime.date(2024, 1, 12))
     assert y_start == datetime.date(2023, 7, 1)
     assert y_end == datetime.date(2024, 6, 30)
     assert q_start == datetime.date(2024, 1, 1)
     assert q_end == datetime.date(2024, 3, 31)

     # April 12 2024 - second quarter of calendar year
     y_start, y_end, q_start, q_end = get_fy_date_ranges(datetime.date(2024, 4, 12))
     assert y_start == datetime.date(2023, 7, 1)
     assert y_end == datetime.date(2024, 6, 30)
     assert q_start == datetime.date(2024, 4, 1)
     assert q_end == datetime.date(2024, 6, 30)

     # July 12 2024 - third quarter of calendar year
     y_start, y_end, q_start, q_end = get_fy_date_ranges(datetime.date(2024, 7, 12))
     assert y_start == datetime.date(2024, 7, 1)
     assert y_end == datetime.date(2025, 6, 30)
     assert q_start == datetime.date(2024, 7, 1)
     assert q_end == datetime.date(2024, 9, 30)

     # October 12 2024 - fourth quarter of calendar year
     y_start, y_end, q_start, q_end = get_fy_date_ranges(datetime.date(2024, 10, 12))
     assert y_start == datetime.date(2024, 7, 1)
     assert y_end == datetime.date(2025, 6, 30)
     assert q_start == datetime.date(2024, 10, 1)
     assert q_end == datetime.date(2024, 12, 31)


def test_calc_fyq(bigger_db_df):
    """
    Test ranges and borehole counts of quarterly and yearly calculations
    """
    df_dict = { 'log1': bigger_db_df }
    y, q = calc_fyq(datetime.date(2022, 5, 6), 'publish_date', df_dict, ['TAS', 'NT'])
    assert y.start == datetime.date(2021, 7, 1)
    assert y.end == datetime.date(2022, 6, 30)
    assert q.start == datetime.date(2022, 4, 1)
    assert q.end == datetime.date(2022, 6, 30)
    # List of BH counts [ Tas counts, NT counts ]
    assert y.cnt_list == [5, 18]
    assert q.cnt_list == [4, 17]
