import os
import sys

# Add in path to source scripts
src_path = os.path.join(os.path.abspath(os.pardir), 'src')
sys.path.insert(0, src_path)

from calculations import calc_bh_depths

from test_db import db_df

#def calc_bh_depths(dfs: dict[str:pd.DataFrame], prov: str, start_date: datetime.date = None,
#                end_date: datetime.date = None, return_cnts: bool = False) -> int:
def test_calc_bh_depths(db_df):
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
    assert depth == 0.975





