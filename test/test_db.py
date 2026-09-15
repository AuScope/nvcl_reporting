import os
import sys
from pathlib import Path

import pytest
import pandas as pd
from psycopg import Connection
from dateutil.relativedelta import relativedelta

from pytest_postgresql import factories


# Add in path to source scripts
src_path = os.path.join(os.path.abspath(os.pardir), "src")
sys.path.insert(0, src_path)

from db.tsg_metadata import TSGMeta

from db.import_db import import_db
from db.export_db import export_db


def test_import_db(tsg_meta_df, small_postgresql_read):
    """ Testing import from sqlite db to dataframe
    """
    dbparams = { 'user': small_postgresql_read.info.user, 
                 'host': small_postgresql_read.info.host, 
                 'port': small_postgresql_read.info.port,
                 'password': 'password' }
    df = import_db(small_postgresql_read.info.dbname, dbparams, "log1", tsg_meta_df)
    assert(isinstance(df, pd.DataFrame))
    assert(len(df) == 20)

def test_import_db_provider_filter(tsg_meta_bigger_df, bigger_postgresql_read):
    """ import_db(provider=...) must return only that provider's rows, and the
        union of per-provider loads must equal the all-providers load.
    """
    dbparams = { 'user': bigger_postgresql_read.info.user,
                 'host': bigger_postgresql_read.info.host,
                 'port': bigger_postgresql_read.info.port,
                 'password': 'password' }
    db_name = bigger_postgresql_read.info.dbname

    full = import_db(db_name, dbparams, "log1", tsg_meta_bigger_df)
    provs = sorted(set(full['provider'].tolist()))
    assert len(provs) >= 2, "fixture should have multiple providers for this test"

    total = 0
    for prov in provs:
        part = import_db(db_name, dbparams, "log1", tsg_meta_bigger_df, provider=prov)
        # Only this provider's rows
        assert set(part['provider'].tolist()) == {prov}
        total += len(part)
    # Per-provider row counts sum to the all-providers count
    assert total == len(full)


def test_update_kms_per_provider_matches_all_at_once(tsg_meta_bigger_df, bigger_postgresql_read):
    """ Core equivalence for the new tactic: computing kms/counts by loading ONE
        provider at a time and assembling the per-year lists must produce exactly
        the same y_list/q_list as loading ALL providers at once (the old way).
    """
    import datetime as _dt
    from calculations import calc_kms4db

    dbparams = { 'user': bigger_postgresql_read.info.user,
                 'host': bigger_postgresql_read.info.host,
                 'port': bigger_postgresql_read.info.port,
                 'password': 'password' }
    db_name = bigger_postgresql_read.info.dbname

    DATA_CATS = ['log0', 'log1', 'log2', 'log3', 'log4', 'log5', 'log6', 'empty', 'nodata']
    date_fieldname = "publish_date"
    report_date = _dt.date(2024, 6, 30)
    REPORT_RANGE = 3

    full_log1 = import_db(db_name, dbparams, "log1", tsg_meta_bigger_df)
    prov_list = sorted(set(full_log1['provider'].tolist()))
    assert len(prov_list) >= 2

    rel_dates = [report_date - relativedelta(years=n) for n in range(REPORT_RANGE)]

    # --- Reference: OLD way, all providers loaded at once ---
    g_all = {}
    for cat in DATA_CATS:
        g_all[cat] = import_db(db_name, dbparams, cat, tsg_meta_bigger_df)
    ref_y, ref_q = [], []
    for rel_date in rel_dates:
        y, q = calc_kms4db(rel_date, date_fieldname, g_all, prov_list)
        ref_y.append(y)
        ref_q.append(q)

    # --- New way: one provider at a time, assemble full-length lists ---
    from types import SimpleNamespace
    y_list = [SimpleNamespace(start=None, end=None,
                              cnt_list=[None]*len(prov_list), kms_list=[None]*len(prov_list))
              for _ in rel_dates]
    q_list = [SimpleNamespace(start=None, end=None,
                              cnt_list=[None]*len(prov_list), kms_list=[None]*len(prov_list))
              for _ in rel_dates]
    for p_idx, prov in enumerate(prov_list):
        g_prov = {cat: import_db(db_name, dbparams, cat, tsg_meta_bigger_df, provider=prov)
                  for cat in DATA_CATS}
        for yr_idx, rel_date in enumerate(rel_dates):
            y, q = calc_kms4db(rel_date, date_fieldname, g_prov, [prov])
            if y_list[yr_idx].start is None:
                y_list[yr_idx].start, y_list[yr_idx].end = y.start, y.end
                q_list[yr_idx].start, q_list[yr_idx].end = q.start, q.end
            y_list[yr_idx].cnt_list[p_idx] = y.cnt_list[0]
            y_list[yr_idx].kms_list[p_idx] = y.kms_list[0]
            q_list[yr_idx].cnt_list[p_idx] = q.cnt_list[0]
            q_list[yr_idx].kms_list[p_idx] = q.kms_list[0]

    # --- Compare ---
    total_kms = 0.0
    for yr_idx in range(len(rel_dates)):
        assert y_list[yr_idx].start == ref_y[yr_idx].start
        assert y_list[yr_idx].end == ref_y[yr_idx].end
        assert q_list[yr_idx].start == ref_q[yr_idx].start
        assert q_list[yr_idx].end == ref_q[yr_idx].end
        for idx in range(len(prov_list)):
            assert y_list[yr_idx].cnt_list[idx] == ref_y[yr_idx].cnt_list[idx]
            assert q_list[yr_idx].cnt_list[idx] == ref_q[yr_idx].cnt_list[idx]
            assert y_list[yr_idx].kms_list[idx] == pytest.approx(ref_y[yr_idx].kms_list[idx])
            assert q_list[yr_idx].kms_list[idx] == pytest.approx(ref_q[yr_idx].kms_list[idx])
            total_kms += ref_y[yr_idx].kms_list[idx]

    assert total_kms > 0.0, "expected non-zero total kms; test would be vacuous otherwise"


def test_export_db(db_df, tsg_meta_df, small_postgresql_read):
    """ Can I export, then re-import and the dataframe is still the same?
    """
    pass
    #dbparams = { 'user': small_postgresql_read.info.user, 
    #             'host': small_postgresql_read.info.host, 
    #             'port': small_postgresql_read.info.port,
    #             'password': 'password' }
    #export_db(db_name, db_params, db_df, "log1", tsg_meta_df)
    #db_df_2 = import_db(os.path.join("test.db"), "log1", tsg_meta_df)
    #assert(db_df_2.compare(db_df).empty)
