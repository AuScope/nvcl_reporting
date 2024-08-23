import os
import sys
import datetime
import math

# Add in path to source scripts
src_path = os.path.join(os.path.abspath(os.pardir), 'src')
sys.path.insert(0, src_path)

from calculations import calc_bh_depths, calc_fyq, get_fy_date_ranges

from test_db import db_df

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
    depth = calc_bh_depths(df_dict, 'TAS')
    assert math.isclose(depth, 0.975)
    cnts, depth = calc_bh_depths(df_dict, 'TAS', return_cnts=True)
    assert math.isclose(depth, 0.975)
    assert cnts == 5


def test_calc_bh_depths_no_cnts(db_df):
    df_dict = { 'log1': db_df }
    depth = calc_bh_depths(df_dict, 'QLD')
    assert math.isclose(depth, 0.0)
    cnts, depth = calc_bh_depths(df_dict, 'QLD', return_cnts=True)
    assert math.isclose(depth, 0.0)
    assert cnts == 0

def test_calc_bh_depths_date_ranges(db_df):
    """ Testing date ranges
    24582|2011-11-06
    24598|2011-11-07
    24725|2014-01-15
    24998|2019-04-02
    24999|2019-07-17
    """
    df_dict = { 'log1': db_df }
    depth = calc_bh_depths(df_dict, 'TAS', start_date=datetime.date(2011,11,12))
    assert math.isclose(depth, 0.962)
    cnts, depth = calc_bh_depths(df_dict, 'TAS', start_date=datetime.date(2011,11,12), return_cnts=True)
    assert math.isclose(depth, 0.962)
    assert cnts == 3

    depth = calc_bh_depths(df_dict, 'TAS', start_date=datetime.date(2011,11,6),
                                           end_date=datetime.date(2011,11,12))
    assert math.isclose(depth, 0.007)
    cnts, depth = calc_bh_depths(df_dict, 'TAS', start_date=datetime.date(2011,11,6),
                                                 end_date=datetime.date(2011,11,12), return_cnts=True)
    assert math.isclose(depth, 0.007)
    assert cnts == 1

    depth = calc_bh_depths(df_dict, 'TAS', end_date=datetime.date(2011,11,7))
    assert math.isclose(depth, 0.006)
    cnts, depth = calc_bh_depths(df_dict, 'TAS', end_date=datetime.date(2011,11,7), return_cnts=True)
    assert math.isclose(depth, 0.006)
    assert cnts == 1


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


#def test_calc_fyq(db_df):
#    """
#    Test quarterly and yearly calculations
#    """
#    df_dict = { 'log1': db_df }
#    y, q = calc_fyq(datetime.date(2012, 1, 4), df_dict, ['TAS'])
#    assert y == 0

