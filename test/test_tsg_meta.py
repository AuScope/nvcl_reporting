import os
import sys
import datetime

# Add in path to source scripts
src_path = os.path.join(os.path.abspath(os.pardir), 'src')
sys.path.insert(0, src_path)

from db.tsg_metadata import TSGMeta

def test_tsg_meta():
    """ Testing TSGMeta class
    """
    t = TSGMeta(os.path.join("data", "metadata.csv"))
    frame = t.get_frame()
    assert frame.shape  == (10,3)
    assert list(frame.columns) == ['nvcl_id', 'hl scan date', 'tsg publish date']
    assert(frame.at[1, 'nvcl_id'] == 'MIN_003542')
    assert(frame.at[8, 'tsg publish date'] == datetime.date(2022, 10, 20))
    assert(frame.at[8, 'hl scan date'] == datetime.date(2022, 6, 1))
