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
    assert t.get_hl_scan_date('yndd013') == datetime.date(2015, 4, 20)
    assert t.get_hl_scan_date('ZNDD005') == datetime.date(2017, 6, 15)
    assert t.get_hl_scan_date('DFGDFHTRJYTRYUJwsdfsdgfsdgsd54353543cewrfww45tKTYKU') == None
    assert len(t.dt_lkup) == 4817
