import sys
import pandas as pd
from datetime import date
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.dialects import postgresql

from db.dbhelpers import make_engine, conv_obj2str
from db.schema import Meas

def export_db(db_name: str, db_params: dict, df: pd.DataFrame, report_category: str, tsg_meta_df: pd.DataFrame):
    engine = make_engine(db_name, db_params)

    rows = []
    for _, row_series in df.iterrows():
        d = row_series.to_dict()
        d["report_category"] = report_category

        d["mincnts"] = conv_obj2str(d["mincnts"])
        d["minerals"] = conv_obj2str(d["minerals"])
        d["data"] = conv_obj2str(d["data"])

        if not isinstance(d["modified_datetime"], date):
            print(f"ERROR: 'modified_datetime' in wrong format: {d['modified_datetime']} in: {d}")
            sys.exit(1)

        d.pop("publish_date", None)
        d.pop("hl_scan_date", None)
        rows.append(d)

    if len(rows) == 0:
        print("No rows inserted")
        return

    with Session(engine) as session:
        stmt = insert(Meas).values(rows)
        
        # Ignore duplicate keys
        stmt = stmt.on_conflict_do_nothing(
            index_elements=[
                "report_category",
                "provider",
                "nvcl_id",
                "log_id",
                "algorithm",
                "log_type",
                "algorithm_id",
            ]
        )
        session.execute(stmt)
        session.commit()
