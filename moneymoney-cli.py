from pysqlcipher3 import dbapi2 as sqlite
import pathlib
import click
import coloredlogs, logging
import prettytable
import datetime
from dataclasses import dataclass
import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.compose import ColumnTransformer
import joblib  


class RelativeDate(click.ParamType):
    name = "relative_date"

    def convert(self, value, param, ctx):
        try:
            if value.endswith("Y"):
                return datetime.datetime.now() - datetime.timedelta(days=int(value[:-1]) * 365)
            elif value.endswith("M"):
                return datetime.datetime.now() - datetime.timedelta(days=int(value[:-1]) * 30)
            elif value.endswith("D"):
                return datetime.datetime.now() - datetime.timedelta(days=int(value[:-1]))
            else:
                return datetime.datetime.strptime(value, "%Y-%m-%d")
        except ValueError:
            self.fail(f"{value} is not a valid date", param, ctx)

class MoneyMoneyDB:

    class Exception (Exception):
        ...
    
    class InvalidKey (Exception):
        ...
    
    

    def __init__ (self,password: str):
        self._logger = logging.getLogger("MoneyMoneyDB")
        self._mm_data_dir = pathlib.Path.home().joinpath("Library/Containers/com.moneymoney-app.retail/Data/Library/Application Support/MoneyMoney/Database").absolute()
        mm_db_path = self._mm_data_dir.joinpath("MoneyMoney.sqlite")
        if not mm_db_path.exists():
            self._logger.error("MoneyMoney database not found!")
            raise FileNotFoundError("MoneyMoney database not found!")   
        self._connection = sqlite.connect(str(mm_db_path))
        self._connection.execute(f"PRAGMA key = '{password}';")
        try:
            self._connection.execute("SELECT COUNT(*) FROM sqlite_master;")
            self._logger.debug("Key is valid, database unlocked!")
        except sqlite.DatabaseError as e:
            self._logger.error("Invalid key, database is not encrypted or it is not a valid DB file")
            raise MoneyMoneyDB.InvalidKey()

    def get_data_dir (self) -> pathlib.Path:
        return self._mm_data_dir
        
    def get_accounts (self):
        cursor = self._connection.cursor()
        cursor.execute("Select rowid, name from accounts;")
        rows = cursor.fetchall()
        for row in rows:
            yield row

    @dataclass
    class Transaction:
        account_key: int
        amount: float
        type: str
        purpose: str
        name: str
        category_key: int
        timestamp: int


    def get_connection (self):
        return self._connection
    


    def get_transactions_query (self,date_from:datetime.datetime,date_to:datetime.datetime,limit_to_accounts,only_uncategorized=False):
        sql = f"select local_account_key, amount, unformatted_type, unformatted_purpose, unformatted_name, category_key, timestamp from transactions where timestamp > {int(date_from.timestamp())} and timestamp < {int(date_to.timestamp())}"
        if only_uncategorized:
            sql = sql + " and category_key = 1"
        if limit_to_accounts is not None and len(limit_to_accounts) > 0:
            sql = sql +  " and local_account_key in ("
            for account in limit_to_accounts:
                sql = sql + str(account) + ","
            sql = sql[:-1] + ")"
        return sql
    

    def get_transactions(self, date_from: datetime.datetime, date_to: datetime.datetime, limit_to_accounts, only_uncategorized=False):
        cursor = self._connection.cursor()
        sql = self.get_transactions_query(date_from, date_to, limit_to_accounts,only_uncategorized)
        cursor.execute(sql)
        while True:
            rows = cursor.fetchmany(100)  # Fetch 100 rows at a time
            if not rows:
                break
            for row in rows:
                yield MoneyMoneyDB.Transaction(*row)

    def get_category_usage (self, date_from: datetime.datetime, date_to: datetime.datetime, limit_to_accounts):
        sql = f"select category_key, count(category_key) as usage from transactions where category_key!=1 and timestamp > {int(date_from.timestamp())} and timestamp < {int(date_to.timestamp())}"
        if limit_to_accounts is not None and len(limit_to_accounts) > 0:
            sql = sql +  " and local_account_key in ("
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
            categories[category_id] = {'name': name, 'group_key': group_key, 'full_name': name}

        cursor.execute("SELECT rowid, name, group_key FROM category_groups")
        rows = cursor.fetchall()
        category_groups = {}
        for row in rows:
            category_id, name, group_key = row
            category_groups[category_id] = {'name': name, 'group_key': group_key}
        
        
        # Build the full category names
        for category_id, category in categories.items():
            full_name = category['name']
            group_key = category['group_key']
            while group_key != 0:
                parent = category_groups.get(group_key)
                if parent:
                    full_name = parent['name'] + '/' + full_name
                    group_key = parent['group_key']
                else:
                    break
            category['full_name'] = full_name
        
        #Return the categories with full names as a dictionary
        return {category_id: category['full_name'] for category_id, category in categories.items()}
        
            



@click.group ()
def cli():
    pass


@cli.command()
@click.option ("--db-password",help="Encryption password of moneymoney DB",required=True)
def list_accounts (db_password):
    db = MoneyMoneyDB (db_password)
    table = prettytable.PrettyTable()
    table.field_names =["ID","Account name"]
    table.align="l"
    for account in db.get_accounts():
        table.add_row (account)
    print (table)


@cli.command()
@click.option("--db-password", help="Encryption password of moneymoney DB", required=True)
@click.option("--date-from", type=RelativeDate(), default="2Y", help="Oldest transaction to be categorized (e.g., 1Y for one year ago or 2021-01-01 for an absolute date)", required=True, show_default=True)
@click.option("--date-to", type=RelativeDate(), default=(datetime.datetime.now() + datetime.timedelta(days=30)).strftime("%Y-%m-%d"), help="Newest transaction to be categorized", required=True, show_default=True)
@click.option("--limit-to-account", help="Limit classification to transactions in the defined account. Can be provided multiple times", multiple=True)
def list_category_usage(db_password, date_from, date_to, limit_to_account):
    """
    List the usage of categories in the transactions.

    Parameters:
    - db_password: Encryption password of the MoneyMoney database.
    - date_from: Oldest transaction to be categorized (e.g., 1Y for one year ago or 2021-01-01 for an absolute date).
    - date_to: Newest transaction to be categorized.
    - limit_to_account: Limit classification to transactions in the defined account. Can be provided multiple times.

    This function lists the usage of categories in the transactions within the specified date range and accounts.
    """
    db = MoneyMoneyDB(db_password)
    category_usage = db.get_category_usage(date_from, date_to, limit_to_account)
    table = prettytable.PrettyTable()
    categories = db.get_categories()
    table.field_names = ["Category ID","Category name","# of transactions"]
    table.align = "l"
    for category_id, usage_count in category_usage.items():
        table.add_row([category_id, categories[category_id], usage_count])
    print(table)




@cli.command()
@click.option("--db-password", help="Encryption password of moneymoney DB", required=True)
@click.option("--date-from", type=RelativeDate(), default="3M", help="Oldest transaction to be categorized (e.g., 1Y for one year ago or 2021-01-01 for an absolute date)", required=True, show_default=True)
@click.option("--date-to", type=RelativeDate(), default=(datetime.datetime.now() + datetime.timedelta(days=30)).strftime("%Y-%m-%d"), help="Newest transaction to be categorized", required=True, show_default=True)
@click.option("--limit-to-account", help="Limit classification to transactions in the defined account. Can be provided multiple times", multiple=True)
@click.option("--model-name", help="Specify the model to be used", required=True, default="default", show_default=True)
@click.option("--apply", help="Apply the categorization to the database", is_flag=True)
def categorize(db_password, date_from, date_to, limit_to_account, model_name, apply):
    """
    Categorize transactions using a trained machine learning model.

    Parameters:
    - db_password: Encryption password of the MoneyMoney database.
    - date_from: Oldest transaction to be categorized (e.g., 1Y for one year ago or 2021-01-01 for an absolute date).
    - date_to: Newest transaction to be categorized.
    - limit_to_account: Limit classification to transactions in the defined account. Can be provided multiple times.
    - model_name: Specify the model to be used.
    - apply: Apply the categorization to the database.

    This function loads transactions from the MoneyMoney database, predicts their categories using a trained model,
    and optionally updates the database with the predicted categories if the --apply flag is specified. It also prints
    the categorized transactions along with their probabilities and a flag indicating whether the probability is above
    the threshold.
    """    
    db = MoneyMoneyDB(db_password)
    transactions = list(db.get_transactions(date_from, date_to, limit_to_account, only_uncategorized=True))

    # Create a DataFrame from the transactions
    df = pd.DataFrame([t.__dict__ for t in transactions])
    if df.empty:
        print("No transactions to categorize.")
        return

    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    df['purpose'] = df['purpose'].fillna('no purpose')
    df['name'] = df['name'].fillna('no name')
    X = df[['amount', 'purpose', 'name','type']]

    # Load the model
    model = joblib.load(str(db.get_data_dir().joinpath(model_name))+".pkl")

    # Predict categories and probabilities
    predicted_categories = model.predict(X)
    predicted_probabilities = model.predict_proba(X).max(axis=1)  

    # Get category names
    category_mapping = db.get_categories()
    category_names = {category_id: name for category_id, name in category_mapping.items()}

    # Apply threshold and update transactions
    threshold = 0.75
    if apply:
        cursor = db.get_connection().cursor()
        for transaction, predicted_category, probabilities in zip(transactions, predicted_categories, predicted_probabilities):
            max_probability = max(probabilities)
            if max_probability > threshold:
                cursor.execute(
                    "UPDATE transactions SET category_key = ? WHERE local_account_key = ? AND timestamp = ?",
                    (predicted_category, transaction.account_key, transaction.timestamp)
                )
        db.get_connection().commit()

    # Print the categorized transactions with probabilities
    df['predicted_category'] = [category_names[category] for category in predicted_categories]
    df['probability'] = predicted_probabilities
    df['above_threshold'] = df['probability'] > threshold
    print(df[['account_key', 'amount', 'purpose', 'name', 'timestamp', 'predicted_category', 'probability', 'above_threshold']])



@cli.command()
@click.option("--db-password", help="Encryption password of moneymoney DB", required=True)
@click.option("--date-from", type=RelativeDate(), default="2Y", help="Oldest transaction to use for training (e.g., -1Y for one year ago or 2021-01-01 for an absolute date)", required=True, show_default=True)
@click.option("--date-to", type=RelativeDate(), default=datetime.datetime.now().strftime("%Y-%m-%d"), help="Newest transaction to use for training (e.g., -3M for three months ago or 2021-01-01 for an absolute date)", required=True, show_default=True)
@click.option("--limit-to-account", help="Limit training to transactions in the defined account. Can be provided multiple times", required=True, multiple=True)
@click.option("--model-name", help="Specify the model name to be created", required=True, default="default", show_default=True)
@click.option("--limit-to-category-file",type=click.File(), help="Provide a text file with a category ID per line to limit the training to those categories", required=False)
def train_model(db_password, date_from, date_to, limit_to_account,model_name,limit_to_category_file):
    db = MoneyMoneyDB(db_password)

    # Prepare dataframe
    df = pd.DataFrame(db.get_transactions(date_from, date_to, limit_to_account))
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')  
    df['purpose'] = df['purpose'].fillna('no purpose')
    df['name'] = df['name'].fillna('no name')

    if limit_to_category_file is not None:
        limit_to_category = [int(line.split()[0]) for line in limit_to_category_file]
        df = df[df['category_key'].isin(limit_to_category)]

    X = df[['amount', 'purpose', 'name','type']]
    y = df['category_key']
    tfidf = TfidfVectorizer(max_features=500)
    scaler = StandardScaler()

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build the pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num_amount', scaler, ['amount']),
            ('text1', tfidf, 'purpose'),
            ('text2', tfidf, 'name'),
            ('text3', tfidf, 'type')
        ]
    )

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)

    # Calculate classification report
    actual_classes = sorted(set(y_test))
    categories = db.get_categories()
    actual_class_names = [categories[cls] for cls in actual_classes]

    print(classification_report(
        y_test,
        y_pred,
        labels=actual_classes,
        target_names=actual_class_names
    ))

    # Save the model
    joblib.dump(model,str(db.get_data_dir().joinpath(model_name))+".pkl")


if __name__ == "__main__":
    cli(obj={})
