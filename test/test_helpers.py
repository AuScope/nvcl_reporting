import os
import sys

import pandas as pd
import pytest

# Add in path to source scripts
src_path = os.path.join(os.path.abspath(os.pardir), 'src')
sys.path.insert(0, src_path)

from helpers import is_known_logid


def test_is_known_logid_present():
    """ Returns True when the log_id is in the known set """
    df = pd.DataFrame({'log_id': ['aaa', 'bbb', 'ccc']})
    assert is_known_logid('bbb', df) is True


def test_is_known_logid_absent():
    """ Returns False when the log_id is not in the known set """
    df = pd.DataFrame({'log_id': ['aaa', 'bbb', 'ccc']})
    assert is_known_logid('zzz', df) is False


def test_is_known_logid_empty_df():
    """ A freshly-created empty DataFrame (no columns) means nothing is known """
    df = pd.DataFrame()
    assert is_known_logid('aaa', df) is False


def test_is_known_logid_missing_column():
    """ A non-empty DataFrame lacking a 'log_id' column means nothing is known """
    df = pd.DataFrame({'other': [1, 2, 3]})
    assert is_known_logid('aaa', df) is False


def test_is_known_logid_none():
    """ None known set is treated as nothing known """
    assert is_known_logid('aaa', None) is False


def test_is_known_logid_after_concat_dedup():
    """ Mirrors how update_data builds known_logid_df: concat of log1 + empty
        log_id columns with dedup. Ids from either source should be recognised.
    """
    log1 = pd.DataFrame({'log_id': ['l1-a', 'l1-b', 'l1-b']})   # note duplicate
    empty = pd.DataFrame({'log_id': ['e-1', 'e-2']})
    known = pd.DataFrame()
    for part in (log1, empty):
        known = pd.concat(
            [known, part.filter(items=['log_id']).drop_duplicates()]
        ).reset_index(drop=True)

    assert is_known_logid('l1-a', known) is True
    assert is_known_logid('l1-b', known) is True
    assert is_known_logid('e-2', known) is True
    assert is_known_logid('missing', known) is False
