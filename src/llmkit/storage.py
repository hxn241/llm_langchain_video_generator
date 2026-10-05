"""Storage: your product's data (rows) in SQLite, a file .state/<name>.sqlite.

    db = storage.connect("expenses")
    db.execute("CREATE TABLE IF NOT EXISTS expenses (...)")
    storage.insert(db, "expenses", {"amount": 12.5, "category": "comida"})   # -> the new row's id
    storage.upsert(db, "reviews", {"id": "r1", "status": "approved"})      # insert, or replace if r1 exists
    storage.select(db, "SELECT * FROM expenses WHERE category = ?", ["comida"])  # -> [dict, ...]

Table: the usual case of a product that processes items and saves one result per item,
where part of the result is a Pydantic object from the model (e.g. "analysis").
Its fields become columns on their own; you only list the other columns.

    REVIEWS = storage.Table("reviews", columns={"id": "TEXT PRIMARY KEY", "text": "TEXT NOT NULL",
                                                "status": "TEXT NOT NULL", "reply": "TEXT"},
                            nested={"analysis": ReviewAnalysis})
    REVIEWS.save([{"id": "r1", "text": "...", "status": "approved", "reply": "...",
                   "analysis": ReviewAnalysis(...) or a dict}])     # insert, or replace by id
    REVIEWS.load()      # -> the same results back, with "analysis" as a dict

SQLite comes with Python (nothing to install) and keeps everything in one file.
For conversations and workflows use persistence.py instead (LangGraph's own format).
"""
import sqlite3
from contextlib import closing

from llmkit import config


def path(name):
    """Where the database `name` lives: .state/<name>.sqlite"""
    return config.STATE_DIR / f"{name}.sqlite"


def connect(name):
    """Opens (or creates) .state/<name>.sqlite. Close it when done, e.g.:
        with closing(storage.connect("news")) as db: ...
    (`with db:` alone only commits; it does not close the file)."""
    file = path(name)
    file.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(file)
    db.row_factory = sqlite3.Row          # rows can be read by column name: row["amount"]
    return db


def insert(db, table, row):
    """Adds a row (a dict {column: value}) and returns its id."""
    columns = ", ".join(row)
    placeholders = ", ".join("?" for _ in row)
    cursor = db.execute(f"INSERT INTO {table} ({columns}) VALUES ({placeholders})", list(row.values()))
    db.commit()
    return cursor.lastrowid


def upsert(db, table, row, key="id"):
    """Adds a row, or REPLACES the columns of the existing row with the same `key`.
    The table must have `key` as PRIMARY KEY (or UNIQUE)."""
    columns = ", ".join(row)
    placeholders = ", ".join("?" for _ in row)
    updates = ", ".join(f"{column} = excluded.{column}" for column in row if column != key)
    db.execute(f"INSERT INTO {table} ({columns}) VALUES ({placeholders}) "
               f"ON CONFLICT({key}) DO UPDATE SET {updates}", list(row.values()))
    db.commit()


def select(db, sql, params=()):
    """Runs a query and returns the rows as dicts."""
    return [dict(row) for row in db.execute(sql, params)]


# ------------------------------------------------------------------- Table
_SQL_TYPES = {int: "INTEGER", float: "REAL", bool: "INTEGER"}      # anything else -> TEXT


class Table:
    """A table of results: one row per item, saved and loaded as dicts.

    name:    the table's name.
    columns: {column: SQL type} of the result's own fields, e.g. {"id": "TEXT PRIMARY KEY"}.
    nested:  {result field: Pydantic class}. That field is stored FLATTENED: each field
             of the class becomes a column (its SQL type comes from the class). On load
             it is rebuilt as a dict (left out if all its columns are empty, e.g. an error).
    db:      the database file .state/<db>.sqlite (default: the table's name).
    key:     the column that identifies each result: saving the same key replaces it.
    """

    def __init__(self, name, columns, nested=None, db=None, key="id"):
        self.name, self.db_name, self.key = name, db or name, key
        self.columns = dict(columns)
        self.nested = {field: list(model.model_fields) for field, model in (nested or {}).items()}
        for model in (nested or {}).values():
            for field, info in model.model_fields.items():
                self.columns.setdefault(field, _SQL_TYPES.get(info.annotation, "TEXT"))

    def connect(self):
        """Opens the database and creates the table the first time. The caller closes it."""
        db = connect(self.db_name)
        definition = ", ".join(f"{column} {sql_type}" for column, sql_type in self.columns.items())
        db.execute(f"CREATE TABLE IF NOT EXISTS {self.name} ({definition})")
        return db

    def save(self, results):
        """Saves the results; one with an existing key replaces it."""
        with closing(self.connect()) as db:
            for result in results:
                row = dict.fromkeys(self.columns)                   # every column, empty by default
                for field, value in result.items():
                    if field in self.nested:                        # flattened into its own columns
                        row.update(value.model_dump() if hasattr(value, "model_dump") else (value or {}))
                    else:
                        row[field] = value
                upsert(db, self.name, row, self.key)

    def load(self, where="", params=()):
        """The saved results, in the order they were first saved.
        where: an optional SQL filter, e.g. load("WHERE status = ?", ["pending_review"])."""
        with closing(self.connect()) as db:
            rows = select(db, f"SELECT * FROM {self.name} {where} ORDER BY rowid", params)
        results = []
        for row in rows:
            nested = {field: {f: row.pop(f) for f in fields} for field, fields in self.nested.items()}
            result = {k: v for k, v in row.items() if v is not None}
            result.update({field: values for field, values in nested.items()
                           if any(v is not None for v in values.values())})
            results.append(result)
        return results