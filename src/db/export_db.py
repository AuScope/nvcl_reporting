import sys
import logging
import pandas as pd
from datetime import date
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.dialects import postgresql

from db.dbhelpers import make_engine, conv_obj2str
from db.schema import Meas

logger = logging.getLogger(__name__)

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
            logger.error("'modified_datetime' in wrong format: %r in: %r", d["modified_datetime"], d)
            sys.exit(1)

        d.pop("publish_date", None)
        d.pop("hl_scan_date", None)
        rows.append(d)

    if len(rows) == 0:
        logger.info("No rows inserted")
        return


    BATCH_SIZE = 1000

    def batched(iterable, n):
        for i in range(0, len(iterable), n):
            logger.info("Inserting rows - %d:%d.", i, i+n)
            sys.stderr.flush()
            yield iterable[i:i+n]

    with Session(engine) as session:
        for chunk in batched(rows, BATCH_SIZE):
            stmt = (
                insert(Meas)
                .values(chunk)
                .on_conflict_do_nothing(
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
            )
            session.execute(stmt)
            session.commit()
