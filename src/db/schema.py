"""
SQLAlchemy ORM version of the Peewee schema.

- Uses SQLAlchemy 2.0 style declarative mappings.
- Keeps JSON stored as TEXT for compatibility with the Peewee version.
- If you're on Postgres and want better querying, consider using JSONB columns instead.
"""

from __future__ import annotations

import json
import math
import sys
from collections import OrderedDict
from dataclasses import dataclass
from datetime import date, datetime
from types import SimpleNamespace
from typing import Any, Iterable, Optional

from sqlalchemy import Boolean, Date, Float, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import TypeDecorator


DATE_FMT = "%Y-%m-%d"

DF_COLUMNS = [
    "provider", "borehole_id", "drill_hole_name", "publish_date", "hl_scan_date",
    "easting", "northing", "crs", "start_depth", "end_depth",
    "has_vnir", "has_swir", "has_tir", "has_mir",
    "nvcl_id", "modified_datetime", "log_id", "algorithm", "log_type", "algorithm_id",
    "minerals", "mincnts", "data",
]


class Base(DeclarativeBase):
    pass


class JSONText(TypeDecorator):
    """
    Stores Python values as JSON in a TEXT column.
    Rough equivalent to your Peewee JSONField.
    """
    impl = Text
    cache_ok = True

    def process_bind_param(self, values: Any, dialect) -> str:
        # Equivalent of db_value()
        if isinstance(values, Iterable) and not isinstance(values, (str, bytes, dict)):
            return json.dumps(list(values))
        if isinstance(values, float) and math.isnan(values):
            return "[]"
        return json.dumps(values)

    def process_result_value(self, value: Optional[str], dialect) -> Any:
        # Equivalent of python_value()
        if value is None:
            return None
        return json.loads(value)


class ScalarsText(TypeDecorator):
    """
    Stores mineral scalars as JSON in a TEXT column.
    """
    impl = Text
    cache_ok = True

    def process_bind_param(self, data: Any, dialect) -> str:
        data_list = []

        if isinstance(data, str):
            return data
        if isinstance(data, OrderedDict):
            for depth, obj in data.items():
                if isinstance(obj, SimpleNamespace):
                    data_list.append([depth, vars(obj)])
                elif isinstance(obj, list) and len(obj) == 0:
                    continue
                else:
                    print(f"ERROR Unknown obj type {type(obj)} in 'data' var: {obj}")
                    sys.exit(1)

        elif data != {} and data != [] and not isinstance(data, list) and not (
            isinstance(data, float) and math.isnan(data)):
            print(f"ERROR Unknown type {type(data)} in 'data' var: {data}")
            sys.exit(1)

        return json.dumps(data_list)

    def process_result_value(self, value: Optional[str], dialect) -> Any:
        if value is None:
            return None
        return json.loads(value)


class Stats(Base):
    """
    Tables of statistics to display
    Composite PK: (stat_name, provider, start_date, end_date)
    """
    __tablename__ = "stats"

    stat_name: Mapped[str] = mapped_column(Text, primary_key=True)
    provider: Mapped[str] = mapped_column(Text, primary_key=True)
    start_date: Mapped[date] = mapped_column(Date, primary_key=True)
    end_date: Mapped[date] = mapped_column(Date, primary_key=True)

    stat_val1: Mapped[float] = mapped_column(Float, nullable=False)
    stat_val2: Mapped[float] = mapped_column(Float, nullable=False)


class Meas(Base):
    """
    Database table structure

    Composite PK:
      (report_category, provider, nvcl_id, log_id, algorithm, log_type, algorithm_id)
    """
    __tablename__ = "meas"

    report_category: Mapped[str] = mapped_column(Text, primary_key=True)
    provider: Mapped[str] = mapped_column(Text, primary_key=True)
    nvcl_id: Mapped[str] = mapped_column(Text, primary_key=True)
    log_id: Mapped[str] = mapped_column(Text, primary_key=True)
    algorithm: Mapped[str] = mapped_column(Text, primary_key=True)
    log_type: Mapped[str] = mapped_column(Text, primary_key=True)
    algorithm_id: Mapped[str] = mapped_column(Text, primary_key=True)

    borehole_id: Mapped[str] = mapped_column(Text, nullable=False)
    drill_hole_name: Mapped[str] = mapped_column(Text, nullable=False)

    # Peewee DoubleField -> SQLAlchemy Float (double precision in Postgres by default)
    easting: Mapped[float] = mapped_column(Float, nullable=False)
    northing: Mapped[float] = mapped_column(Float, nullable=False)

    crs: Mapped[str] = mapped_column(Text, nullable=False)  # e.g. "EPSG:7842"

    start_depth: Mapped[float] = mapped_column(Float, nullable=False)
    end_depth: Mapped[float] = mapped_column(Float, nullable=False)

    has_vnir: Mapped[bool] = mapped_column(Boolean, nullable=False)
    has_swir: Mapped[bool] = mapped_column(Boolean, nullable=False)
    has_tir: Mapped[bool] = mapped_column(Boolean, nullable=False)
    has_mir: Mapped[bool] = mapped_column(Boolean, nullable=False)

    # Note: your DF_COLUMNS includes has_mir, but Meas has has_mir in Peewee too.
    # If you also have has_tir/has_mir/has_mir exactly, keep consistent.

    modified_datetime: Mapped[Optional[date]] = mapped_column(Date, nullable=True)

    minerals: Mapped[Any] = mapped_column(JSONText, nullable=False)
    mincnts: Mapped[Any] = mapped_column(JSONText, nullable=False)
    data: Mapped[Any] = mapped_column(ScalarsText, nullable=False)


# Equivalent of: DB_COLUMNS = [field.name for field in Meas._meta.fields.values()]
DB_COLUMNS = [c.name for c in Meas.__table__.columns]


@dataclass
class DF_Row:
    provider: str
    borehole_id: str
    drill_hole_name: str
    publish_date: date
    hl_scan_date: date
    easting: float
    northing: float
    crs: str
    start_depth: float
    end_depth: float
    has_vnir: bool
    has_swir: bool
    has_tir: bool
    has_mir: bool
    nvcl_id: str
    modified_datetime: datetime
    log_id: str
    algorithm: str
    log_type: str
    algorithm_id: str
    minerals: list
    mincnts: dict
    data: SimpleNamespace

    def as_list(self):
        ATTR_LIST = [
            "provider", "borehole_id", "drill_hole_name", "publish_date", "hl_scan_date",
            "easting", "northing", "crs", "start_depth", "end_depth",
            "has_vnir", "has_swir", "has_tir", "has_mir",
            "nvcl_id", "modified_datetime", "log_id", "algorithm", "log_type", "algorithm_id",
            "minerals", "mincnts", "data",
        ]
        return [getattr(self, attr) for attr in ATTR_LIST]
