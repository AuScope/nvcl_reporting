import sys
from sqlalchemy import delete
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from db.dbhelpers import make_engine
from db.schema import Stats

def export_kms(db_name: str, db_params: dict, prov_list: list, y_list, q_list):
    print(f"Opening: {db_name}")
    engine = make_engine(db_name, db_params)

    rows = []
    for idx, prov in enumerate(prov_list):
        for y in y_list:
            rows.append({
                "stat_name": "borehole_cnt_kms",
                "provider": prov,
                "start_date": y.start,
                "end_date": y.end,
                "stat_val1": y.cnt_list[idx],
                "stat_val2": y.kms_list[idx],
            })
        for q in q_list:
            rows.append({
                "stat_name": "borehole_cnt_kms",
                "provider": prov,
                "start_date": q.start,
                "end_date": q.end,
                "stat_val1": q.cnt_list[idx],
                "stat_val2": q.kms_list[idx],
            })

    with Session(engine) as session:
        try:
            session.execute(delete(Stats))  # wipe table (like your else branch)
            session.bulk_insert_mappings(Stats, rows)
            session.commit()
        except IntegrityError as e:
            session.rollback()
            print(f"Duplicate row error: {e}")
            print("Tried to insert", rows)
            sys.exit(1)
        except Exception as e:
            session.rollback()
            print(f"Bad param error: {e}")
            print("Tried to insert", rows)
            sys.exit(1)
