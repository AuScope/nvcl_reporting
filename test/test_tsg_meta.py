import os
import sys
import datetime

# Add in path to source scripts
src_path = os.path.join(os.path.abspath(os.pardir), 'src')
sys.path.insert(0, src_path)

my_dir = os.path.dirname(os.path.abspath(__file__))

from db.tsg_metadata import TSGMeta

def test_tsg_meta():
    """ Testing TSGMeta class
    """
    t = TSGMeta(os.path.join(my_dir, "data", "small_metadata.csv"), False)
    frame = t.get_frame()
    assert frame.shape  == (5,3)
    assert list(frame.columns) == ['nvcl_id', 'hl scan date', 'tsg publish date']
    assert(frame.at[1, 'nvcl_id'] == 'Mutineer 3')
    assert(frame.at[2, 'tsg publish date'] == datetime.date(2022, 6, 27))
    assert(frame.at[4, 'hl scan date'] == datetime.date(2019, 7, 17))
