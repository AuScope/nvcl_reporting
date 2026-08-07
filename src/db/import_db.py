import pandas as pd
import pandas
import sqlalchemy
from sqlalchemy import text, inspect
import psycopg2

from db.dbhelpers import make_engine, conv_str2dt, conv_str2json, db_col_str
from db.schema import Base, DF_COLUMNS

def import_db(db_name: str, db_params: dict, report_datacat: str, tsg_meta_df: pd.DataFrame) -> pd.DataFrame:
    engine = make_engine(db_name, db_params)

    sql = text(f"SELECT {db_col_str()} FROM meas WHERE report_category = :cat")

    with engine.connect() as conn:
        try:
            src_df = pd.read_sql(sql, conn, params={"cat": report_datacat})
        except sqlalchemy.exc.ProgrammingError as pe:
            print(f"Cannot find data in database.")
            src_df = pd.DataFrame(columns=DF_COLUMNS)
            # Create tables
            insp = inspect(engine)
            if "meas" not in insp.get_table_names():
                print("Creating tables")
                Base.metadata.create_all(engine)


    assert type(src_df.get("modified_datetime")) is not pd.Timestamp

    new_df = pd.DataFrame(columns=DF_COLUMNS).drop(columns=["publish_date", "hl_scan_date"])

    for col in src_df.columns:
        #if col in ["modified_datetime"]:
        #    new_df[col] = src_df[col].apply(conv_str2dt)
        if col in ["minerals", "mincnts", "data"]:
            new_df[col] = src_df[col].apply(conv_str2json)
        elif col in ["publish_date", "hl_scan_date"]:
            continue
        else:
            new_df[col] = src_df[col]

    if not new_df.empty:
        merged_df = pd.merge(new_df, tsg_meta_df, left_on="nvcl_id", right_on="nvcl_id")
        merged_df = merged_df.rename(columns={"hl scan date": "hl_scan_date", "tsg publish date": "publish_date"})
        return merged_df

    return pd.DataFrame(columns=DF_COLUMNS)
