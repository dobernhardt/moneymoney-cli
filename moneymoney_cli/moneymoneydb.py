import datetime
import pathlib
from dataclasses import dataclass
from pysqlcipher3 import dbapi2 as sqlite
from sqlite3 import OperationalError, DatabaseError
from .console import console

class MoneyMoneyDB:

    class Exception(Exception):
        ...

    @staticmethod
    def get_data_dir() -> pathlib.Path:
        return pathlib.Path.home().joinpath("Library/Containers/com.moneymoney-app.retail/Data/Library/Application Support/MoneyMoney/Database").absolute()

    def __init__(self, password: str):
        mm_db_path = MoneyMoneyDB.get_data_dir().joinpath("MoneyMoney.sqlite")
        console.log ("opening MoneyMoney database at", mm_db_path, style="dim")
        if not mm_db_path.exists():
            console.log("MoneyMoney database not found!",style="red")
            raise FileNotFoundError("MoneyMoney database not found!")
        uri = f"file:{mm_db_path}?mode=ro"
        self._connection = sqlite.connect(uri, uri=True)
        self._connection.execute(f"PRAGMA key = '{password}';")
        try:
            self._connection.execute("SELECT COUNT(*) FROM sqlite_master;")
            console.log("Key is valid, database unlocked!",style="green")
        except sqlite.OperationalError as e:
            console.log (f"Database error: {e}. Propably database is open in MoneyMoney",style="red")
            raise MoneyMoneyDB.Exception()
        except sqlite.DatabaseError as e:
            console.log (f"Database error {e}. Invalid key, database is not encrypted or it is not a valid DB file",style="red")
            raise MoneyMoneyDB.Exception()


    def get_accounts(self):
        cursor = self._connection.cursor()
        cursor.execute("Select rowid, name from accounts;")
        rows = cursor.fetchall()
        for row in rows:
            yield row

    @dataclass
    class Transaction:
        transaction_id: int
        account_key: int
        amount: float
        type: str
        purpose: str
        name: str
        category_key: int
        timestamp: int

    def get_connection(self):
        return self._connection

    def get_transactions_query(self, date_from: datetime.datetime, date_to: datetime.datetime, limit_to_accounts, only_uncategorized=False):
        sql = (
            f"SELECT rowid, local_account_key, amount, unformatted_type, unformatted_purpose, unformatted_name, category_key, "
            f"timestamp FROM transactions WHERE timestamp > {int(date_from.timestamp())} AND timestamp < {int(date_to.timestamp())}"
        )
        if only_uncategorized:
            sql = sql + " and category_key = 1"
        if limit_to_accounts is not None and len(limit_to_accounts) > 0:
            sql = sql + " and local_account_key in ("
            for account in limit_to_accounts:
                sql = sql + str(account) + ","
            sql = sql[:-1] + ")"
        return sql

    def get_transactions(self, date_from: datetime.datetime, date_to: datetime.datetime, limit_to_accounts, only_uncategorized=False):
        cursor = self._connection.cursor()
        sql = self.get_transactions_query(date_from, date_to, limit_to_accounts, only_uncategorized)
        cursor.execute(sql)
        while True:
            rows = cursor.fetchmany(100)  # Fetch 100 rows at a time
            if not rows:
                break
            for row in rows:
                yield MoneyMoneyDB.Transaction(*row)

    def get_category_usage(self, date_from: datetime.datetime, date_to: datetime.datetime, limit_to_accounts):
        sql = (
            f"select category_key, count(category_key) as usage from transactions where category_key!=1 "
            f"and timestamp > {int(date_from.timestamp())} and timestamp < {int(date_to.timestamp())}"
        )
        if limit_to_accounts is not None and len(limit_to_accounts) > 0:
            sql = sql + " and local_account_key in ("
            for account in limit_to_accounts:
                sql = sql + str(account) + ","
            sql = sql[:-1] + ")"
        sql = sql + " group by category_key order by usage desc"
        cursor = self._connection.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        return {item[0]: item[1] for item in rows}

    def get_categories(self):
        cursor = self._connection.cursor()

        # Fetch all categories and category groups
        cursor.execute("SELECT rowid, name, group_key FROM categories")
        rows = cursor.fetchall()
        categories = {}
        for row in rows:
            category_id, name, group_key = row
            categories[category_id] = {"name": name, "group_key": group_key, "full_name": name}

        cursor.execute("SELECT rowid, name, group_key FROM category_groups")
        rows = cursor.fetchall()
        category_groups = {}
        for row in rows:
            category_id, name, group_key = row
            category_groups[category_id] = {"name": name, "group_key": group_key}

        # Build the full category names
        for category_id, category in categories.items():
            full_name = category["name"]
            group_key = category["group_key"]
            while group_key != 0:
                parent = category_groups.get(group_key)
                if parent:
                    full_name = parent["name"] + "/" + full_name
                    group_key = parent["group_key"]
                else:
                    break
            category["full_name"] = full_name

        # Return the categories with full names as a dictionary
        return {category_id: category["full_name"] for category_id, category in categories.items()}
