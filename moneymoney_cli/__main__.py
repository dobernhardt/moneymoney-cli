import datetime

import click

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .moneymoneydb import MoneyMoneyDB
from .relative_date import RelativeDate
from .word_vec import Word2VecVectorizer
from rich.console import Console
from rich.table import Table


@click.group()
def cli():
    pass


@cli.command()
@click.option("--db-password", help="Encryption password of moneymoney DB", required=True)
def list_accounts(db_password):
    db = MoneyMoneyDB(db_password)
    console = Console()
    table = Table(title="Accounts")

    table.add_column("ID", justify="left", style="cyan", no_wrap=True)
    table.add_column("Account name", justify="left", style="magenta")

    for account in db.get_accounts():
        table.add_row(str(account[0]), account[1])

    console.print(table)


@cli.command()
# fmt: off
@click.option("--db-password", help="Encryption password of moneymoney DB", required=True)
@click.option("--date-from", type=RelativeDate(), default="2Y", help="Oldest transaction to be categorized (e.g., 1Y for one year ago or 2021-01-01 for an absolute date)", required=True, show_default=True)  # noqa: E501
@click.option("--date-to", type=RelativeDate(), default=(datetime.datetime.now() + datetime.timedelta(days=30)).strftime("%Y-%m-%d"), help="Newest transaction to be categorized", required=True, show_default=True)  # noqa: E501
@click.option("--limit-to-account", help="Limit classification to transactions in the defined account. Can be provided multiple times", multiple=True)
# fmt: on
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
    categories = db.get_categories()

    console = Console()
    table = Table(title="Category Usage")

    table.add_column("Category ID", justify="left", style="cyan", no_wrap=True)
    table.add_column("Category name", justify="left", style="magenta")
    table.add_column("# of transactions", justify="left", style="green")

    for category_id, usage_count in category_usage.items():
        table.add_row(str(category_id), categories[category_id], str(usage_count))

    console.print(table)


@cli.command()
# fmt: off
@click.option("--db-password", help="Encryption password of moneymoney DB", required=True)
@click.option("--date-from", type=RelativeDate(), default="3M", help="Oldest transaction to be categorized (e.g., 1Y for one year ago or 2021-01-01 for an absolute date)", required=True, show_default=True)  # noqa: E501
@click.option("--date-to", type=RelativeDate(), default=(datetime.datetime.now() + datetime.timedelta(days=30)).strftime("%Y-%m-%d"), help="Newest transaction to be categorized", required=True, show_default=True)  # noqa: E501
@click.option("--limit-to-account", help="Limit classification to transactions in the defined account. Can be provided multiple times", multiple=True)
@click.option("--model-name", help="Specify the model to be used", required=True, default="default", show_default=True)
@click.option("--propability-threshold", help="Specify the prediction threshold. Only predictions with a higher propability are conciders", required=True, default=0.75, show_default=True)   # noqa: E501
@click.option("--apply", help="Apply the categorization to the database", is_flag=True)
@click.option("--overwrite", help="Categorize also already categorized transactions", is_flag=True)
# fmt: on
def categorize(db_password, date_from, date_to, limit_to_account, model_name, apply, propability_threshold, overwrite):
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
    transactions = list(db.get_transactions(date_from, date_to, limit_to_account, only_uncategorized=(not overwrite)))

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
    console = Console()
    table = Table(title="Categorized Transactions")

    table.add_column("Purpose", justify="left")
    table.add_column("Name", justify="left")
    table.add_column("Amount", justify="right")
    table.add_column("Predicted Category", justify="left")
    table.add_column("Probability", justify="right")

    for _, row in df.iterrows():
        probability_style = "bright_black" if row["probability"] < propability_threshold else "white"
        table.add_row(
            row["purpose"][:50],
            row["name"][:50],
            f"{row['amount']:.2f}",
            row["predicted_category"],
            f"{row['probability']:.2f}",
            style=probability_style
        )
    console.print(table)

    # Apply threshold and update transactions
    if apply:
        df = df[df["probability"] >= propability_threshold]
        cursor = db.get_connection().cursor()
        for transaction_id, predicted_category_id, probabilities in df[["transaction_id", "predicted_category_id", "probability"]].values:
            cursor.execute("UPDATE transactions SET category_key = ? WHERE rowid = ?", (int(predicted_category_id), int(transaction_id)))
        db.get_connection().commit()


@cli.command()
# fmt: off
@click.option("--db-password", help="Encryption password of moneymoney DB", required=True)
@click.option("--date-from", type=RelativeDate(), default="2Y", help="Oldest transaction to use for training (e.g., -1Y for one year ago or 2021-01-01 for an absolute date)", required=True, show_default=True)  # noqa: E501
@click.option("--date-to", type=RelativeDate(), default=datetime.datetime.now().strftime("%Y-%m-%d"), help="Newest transaction to use for training (e.g., -3M for three months ago or 2021-01-01 for an absolute date)", required=True, show_default=True)  # noqa: E501
@click.option("--limit-to-account", help="Limit training to transactions in the defined account. Can be provided multiple times", required=True, multiple=True)
@click.option("--model-name", help="Specify the model name to be created", required=True, default="default", show_default=True)
@click.option("--limit-to-category-file", type=click.File(), help="Provide a text file with a category ID per line to limit the training to those categories", required=False)  # noqa: E501
# fmt: on
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
