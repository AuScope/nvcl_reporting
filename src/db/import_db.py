import gc
import logging
import pandas as pd
import pandas
import sqlalchemy
from sqlalchemy import text, inspect
import psycopg2

from db.dbhelpers import make_engine, conv_str2dt, conv_str2json, db_col_str, JSON_COLS
from db.schema import Base, DF_COLUMNS

logger = logging.getLogger(__name__)

def import_db(db_name: str, db_params: dict, report_datacat: str, tsg_meta_df: pd.DataFrame,
              load_json: bool = True, provider: str = None) -> pd.DataFrame:
    """
    Import a report category from the DB into a merged DataFrame.

    :param load_json: when False, the heavy JSON columns (minerals, mincnts, data)
        are neither fetched nor parsed. They are returned as empty columns so the
        DataFrame schema (DF_COLUMNS) is preserved. Use for lightweight consumers.
    :param provider: when given, only rows for this provider are fetched. Lets
        callers load and process one provider at a time to bound memory.
    """
    engine = make_engine(db_name, db_params)

    try:
        where = "WHERE report_category = :cat"
        params = {"cat": report_datacat}
        if provider is not None:
            where += " AND provider = :prov"
            params["prov"] = provider
        sql = text(f"SELECT {db_col_str(include_json=load_json)} FROM meas {where}")

        logger.info(f"Fetching from DB using {sql}")

        with engine.connect() as conn:
            try:
                src_df = pd.read_sql(sql, conn, params=params)
            except sqlalchemy.exc.ProgrammingError as pe:
                logger.warning("Cannot find data in database.")
                src_df = pd.DataFrame(columns=DF_COLUMNS)
                # Create tables
                insp = inspect(engine)
                if "meas" not in insp.get_table_names():
                    logger.info("Creating tables")
                    Base.metadata.create_all(engine)

        logger.info(f"DONE! Fetched data for {report_datacat}")

        assert type(src_df.get("modified_datetime")) is not pd.Timestamp

        # Drop date columns
        src_df = src_df.drop(columns=["publish_date", "hl_scan_date"], errors="ignore")

        # Convert JSON columns in-situ (only present when load_json=True)
        if load_json:
            for col in src_df.columns:
                #if col in ["modified_datetime"]:
                #    src_df[col] = src_df[col].apply(conv_str2dt)
                if col in JSON_COLS:
                    src_df[col] = src_df[col].apply(conv_str2json)

        logger.info(f"Converted to 'src_df'")

        if not src_df.empty:
            merged_df = pd.merge(src_df, tsg_meta_df, left_on="nvcl_id", right_on="nvcl_id")
            merged_df = merged_df.rename(columns={"hl scan date": "hl_scan_date", "tsg publish date": "publish_date"})
            # If JSON columns were skipped, add them back empty so schema == DF_COLUMNS
            if not load_json:
                for col in JSON_COLS:
                    if col not in merged_df.columns:
                        merged_df[col] = [[] for _ in range(len(merged_df))]
            logger.info(f"Merging done, returning")
            return merged_df

        logger.info("Returning empty df")
        return pd.DataFrame(columns=DF_COLUMNS)
    finally:
        engine.dispose()
