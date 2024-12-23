import datetime

from .moneymoneydb import MoneyMoneyDB

import click
import gensim.downloader as api
import joblib
import numpy as np
import pandas as pd
import prettytable

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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


@click.group()
def cli():
    pass


@cli.command()
@click.option("--db-password", help="Encryption password of moneymoney DB", required=True)
def list_accounts(db_password):
    db = MoneyMoneyDB(db_password)
    table = prettytable.PrettyTable()
    table.field_names = ["ID", "Account name"]
    table.align = "l"
    for account in db.get_accounts():
        table.add_row(account)
    print(table)


@cli.command()
@click.option("--db-password", help="Encryption password of moneymoney DB", required=True)
@click.option(
    "--date-from",
    type=RelativeDate(),
    default="2Y",
    help="Oldest transaction to be categorized (e.g., 1Y for one year ago or 2021-01-01 for an absolute date)",
    required=True,
    show_default=True,
)
@click.option(
    "--date-to",
    type=RelativeDate(),
    default=(datetime.datetime.now() + datetime.timedelta(days=30)).strftime("%Y-%m-%d"),
    help="Newest transaction to be categorized",
    required=True,
    show_default=True,
)
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
    table.field_names = ["Category ID", "Category name", "# of transactions"]
    table.align = "l"
    for category_id, usage_count in category_usage.items():
        table.add_row([category_id, categories[category_id], usage_count])
    print(table)


@cli.command()
@click.option("--db-password", help="Encryption password of moneymoney DB", required=True)
@click.option(
    "--date-from",
    type=RelativeDate(),
    default="3M",
    help="Oldest transaction to be categorized (e.g., 1Y for one year ago or 2021-01-01 for an absolute date)",
    required=True,
    show_default=True,
)
@click.option(
    "--date-to",
    type=RelativeDate(),
    default=(datetime.datetime.now() + datetime.timedelta(days=30)).strftime("%Y-%m-%d"),
    help="Newest transaction to be categorized",
    required=True,
    show_default=True,
)
@click.option("--limit-to-account", help="Limit classification to transactions in the defined account. Can be provided multiple times", multiple=True)
@click.option("--model-name", help="Specify the model to be used", required=True, default="default", show_default=True)
@click.option(
    "--propability-threshold",
    help="Specify the prediction threshold. Only predictions with a higher propability are conciders",
    required=True,
    default=0.75,
    show_default=True,
)
@click.option("--apply", help="Apply the categorization to the database", is_flag=True)
def categorize(db_password, date_from, date_to, limit_to_account, model_name, apply, propability_threshold):
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
    and optionally updates the database with the predicted categories if the --apply flag is specified.
    """
    db = MoneyMoneyDB(db_password)
    transactions = list(db.get_transactions(date_from, date_to, limit_to_account, only_uncategorized=True))

    # Create a DataFrame from the transactions
    df = pd.DataFrame([t.__dict__ for t in transactions])
    if df.empty:
        print("No transactions to categorize.")
        return

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["purpose"] = df["purpose"].fillna("no purpose")
    df["name"] = df["name"].fillna("no name")
    X = df[["amount", "purpose", "name", "type"]]

    # Load the model
    model = joblib.load(str(db.get_data_dir().joinpath(model_name)) + ".pkl")

    # Predict categories and probabilities
    predicted_categories = model.predict(X)
    predicted_probabilities = model.predict_proba(X).max(axis=1)

    # Get category names
    category_mapping = db.get_categories()
    category_names = {category_id: name for category_id, name in category_mapping.items()}

    df["predicted_category_id"] = predicted_categories
    df["predicted_category"] = [category_names[category] for category in predicted_categories]
    df["probability"] = predicted_probabilities

    # Print the categorized transactions with probabilities
    print(df[["transaction_id", "purpose", "name", "amount", "predicted_category", "probability"]].to_string(index=False))

    # Apply threshold and update transactions
    if apply:
        cursor = db.get_connection().cursor()
        for transaction_id, predicted_category_id, probabilities in df[["transaction_id", "predicted_category_id", "probability"]].values:
            if probabilities > propability_threshold:
                cursor.execute("UPDATE transactions SET category_key = ? WHERE rowid = ?", (int(predicted_category_id), int(transaction_id)))
        db.get_connection().commit()


class Word2VecVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name="glove-wiki-gigaword-50"):
        self.model_name = model_name
        self.word2vec = api.load(model_name)
        self.dim = self.word2vec.vector_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([self._get_sentence_vector(sentence) for sentence in X])

    def _get_sentence_vector(self, sentence):
        words = sentence.split()
        word_vectors = [self.word2vec[word] for word in words if word in self.word2vec]
        if len(word_vectors) == 0:
            return np.zeros(self.dim)
        return np.mean(word_vectors, axis=0)


# Modify the train_model function
@cli.command()
@click.option("--db-password", help="Encryption password of moneymoney DB", required=True)
@click.option(
    "--date-from",
    type=RelativeDate(),
    default="2Y",
    help="Oldest transaction to use for training (e.g., -1Y for one year ago or 2021-01-01 for an absolute date)",
    required=True,
    show_default=True,
)
@click.option(
    "--date-to",
    type=RelativeDate(),
    default=datetime.datetime.now().strftime("%Y-%m-%d"),
    help="Newest transaction to use for training (e.g., -3M for three months ago or 2021-01-01 for an absolute date)",
    required=True,
    show_default=True,
)
@click.option("--limit-to-account", help="Limit training to transactions in the defined account. Can be provided multiple times", required=True, multiple=True)
@click.option("--model-name", help="Specify the model name to be created", required=True, default="default", show_default=True)
@click.option(
    "--limit-to-category-file",
    type=click.File(),
    help="Provide a text file with a category ID per line to limit the training to those categories",
    required=False,
)
def train_model(db_password, date_from, date_to, limit_to_account, model_name, limit_to_category_file):
    db = MoneyMoneyDB(db_password)

    # Prepare dataframe
    df = pd.DataFrame(db.get_transactions(date_from, date_to, limit_to_account))
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["purpose"] = df["purpose"].fillna("no purpose")
    df["name"] = df["name"].fillna("no name")
    df["type"] = df["type"].fillna("no type")

    if limit_to_category_file is not None:
        limit_to_category = [int(line.split()[0]) for line in limit_to_category_file]
        df = df[df["category_key"].isin(limit_to_category)]

    X = df[["amount", "purpose", "name", "type"]]
    y = df["category_key"]
    scaler = StandardScaler()

    # Use Word2Vec for text fields
    word2vec_vectorizer = Word2VecVectorizer()

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build the pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num_amount", scaler, ["amount"]),
            ("text1", word2vec_vectorizer, "purpose"),
            ("text2", word2vec_vectorizer, "name"),
            ("text3", word2vec_vectorizer, "type"),
        ]
    )

    model = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))])

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)

    # Calculate classification report
    actual_classes = sorted(set(y_test))
    categories = db.get_categories()
    actual_class_names = [categories[cls] for cls in actual_classes]

    print(classification_report(y_test, y_pred, labels=actual_classes, target_names=actual_class_names))

    # Save the model
    joblib.dump(model, str(db.get_data_dir().joinpath(model_name)) + ".pkl")


if __name__ == "__main__":
    cli(obj={})
