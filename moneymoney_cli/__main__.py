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
import yaml
import pathlib
from .console import console
from .config import read_config


@click.group()
def cli():
    pass


@cli.command()
@click.option("--db-password", help="Encryption password of moneymoney DB")
def list_accounts(db_password):
    config = read_config()
    db_password = db_password if db_password is not None else config.get("db_password")
    if None in [db_password]:
        raise click.UsageError("Please provide all required parameters")
    try:
        db = MoneyMoneyDB(db_password)
    except MoneyMoneyDB.Exception:
        console.log("Error opening database", style="red")
        return
    table = Table(title="Accounts")

    table.add_column("ID", justify="left", style="cyan", no_wrap=True)
    table.add_column("Account name", justify="left", style="magenta")

    for account in db.get_accounts():
        table.add_row(str(account[0]), account[1])

    console.print(table)


@cli.command()
# fmt: off
@click.option("--config-profile",help="Configuration profile to use to read config values from instead of specifying on the command line", default="default", show_default=True, required=False)
@click.option("--db-password", help="Encryption password of moneymoney DB")
@click.option("--date-from", type=RelativeDate(), default="2Y", help="Oldest transaction to be categorized (e.g., 1Y for one year ago or 2021-01-01 for an absolute date)", required=True, show_default=True)  # noqa: E501
@click.option("--date-to", type=RelativeDate(), default=(datetime.datetime.now() + datetime.timedelta(days=30)).strftime("%Y-%m-%d"), help="Newest transaction to be categorized", required=True, show_default=True)  # noqa: E501
@click.option("--limit-to-account", help="Limit classification to transactions in the defined account. Can be provided multiple times", multiple=True)
# fmt: on
def list_category_usage(config_profile,db_password, date_from, date_to, limit_to_account):
    """
    List the usage of categories in the transactions.

    Parameters:
    - db_password: Encryption password of the MoneyMoney database.
    - date_from: Oldest transaction to be categorized (e.g., 1Y for one year ago or 2021-01-01 for an absolute date).
    - date_to: Newest transaction to be categorized.
    - limit_to_account: Limit classification to transactions in the defined account. Can be provided multiple times.

    This function lists the usage of categories in the transactions within the specified date range and accounts.
    """
    config = read_config(config_profile)
    db_password = db_password if db_password is not None else config.get("db_password")
    if None in [db_password]:
        raise click.UsageError("Please provide all required parameters")
    try:
        db = MoneyMoneyDB(db_password)
    except MoneyMoneyDB.Exception:
        console.log("Error opening database", style="red")
        return
    category_usage = db.get_category_usage(date_from, date_to, limit_to_account)
    categories = db.get_categories()

    table = Table(title="Category Usage")

    table.add_column("Category ID", justify="left", style="cyan", no_wrap=True)
    table.add_column("Category name", justify="left", style="magenta")
    table.add_column("# of transactions", justify="left", style="green")

    for category_id, usage_count in category_usage.items():
        table.add_row(str(category_id), categories[category_id], str(usage_count))

    console.print(table)


@cli.command()
# fmt: off
@click.option("--db-password", help="Encryption password of moneymoney DB")
@click.option("--config-profile",help="Configuration profile to use to read config values from instead of specifying on the command line", default="default", show_default=True, required=False)
@click.option("--date-from", type=RelativeDate(), default="3M", help="Oldest transaction to be categorized (e.g., 1Y for one year ago or 2021-01-01 for an absolute date)", required=True, show_default=True)  # noqa: E501
@click.option("--date-to", type=RelativeDate(), default=(datetime.datetime.now() + datetime.timedelta(days=30)).strftime("%Y-%m-%d"), help="Newest transaction to be categorized", required=True, show_default=True)  # noqa: E501
@click.option("--limit-to-account", help="Limit classification to transactions in the defined account. Can be provided multiple times", multiple=True)
@click.option("--model-name", help="Specify the model to be used",default="default", show_default=True)
@click.option("--propability-threshold", help="Specify the prediction threshold. Only predictions with a higher propability are conciders", required=True, default=0.75, show_default=True)   # noqa: E501
@click.option("--apply", help="Apply the categorization to the database", is_flag=True)
@click.option("--overwrite", help="Categorize also already categorized transactions", is_flag=True)
# fmt: on
def categorize(db_password, date_from, date_to, limit_to_account, model_name, apply, propability_threshold, overwrite,config_profile):
    """
    Categorize transactions using a trained machine learning model.
    Parameters can either be provided as command line arguments or read from a configuration file. If a config.yml file is either found in the current directory
    or in the MoneyMoney data directory it will be read and configuraiton from the specified profile will be used. Any parameters provided as command line arguments will override the configuration file values.

    """
    config = read_config(config_profile)
    db_password = db_password if db_password is not None else config.get("db_password")
    limit_to_account = limit_to_account if limit_to_account else config.get("limit_to_accounts")
    model_name = model_name if model_name is not None else config.get("model_name")
    propability_threshold = propability_threshold if propability_threshold is not None else config.get("propability_threshold")
    date_from = date_from if date_from is not None else config.get("date_from")
    date_to = date_to if date_to is not None else config.get("date_to")
    if None in [db_password, model_name, propability_threshold, date_from, date_to]:
        raise click.UsageError("Please provide all required parameters")
    try:
        db = MoneyMoneyDB(db_password)
    except MoneyMoneyDB.Exception:
        console.log("Error opening database", style="red")
        return
    console.log("Reading transactions from database...")
    transactions = list(db.get_transactions(date_from, date_to, limit_to_account, only_uncategorized=(not overwrite)))

    # Create a DataFrame from the transactions
    df = pd.DataFrame([t.__dict__ for t in transactions])
    if df.empty:
        console.log("No transactions to categorize.",style="red")
        return

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["purpose"] = df["purpose"].fillna("no purpose")
    df["name"] = df["name"].fillna("no name")
    X = df[["amount", "purpose", "name", "type"]]

    # Load the model
    model = joblib.load(str(db.get_data_dir().joinpath(model_name)) + ".pkl")

    # Predict categories and probabilities
    console.log("Predicting categories...")
    predicted_categories = model.predict(X)
    predicted_probabilities = model.predict_proba(X).max(axis=1)

    # Get category names
    category_mapping = db.get_categories()
    category_names = {category_id: name for category_id, name in category_mapping.items()}

    df["predicted_category_id"] = predicted_categories
    df["predicted_category"] = [category_names[category] for category in predicted_categories]
    df["probability"] = predicted_probabilities

    # Print the categorized transactions with probabilities
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
    df_filtered = df[df["probability"] >= propability_threshold]
    if apply:
        console.log("Applying categorization to database...")
        cursor = db.get_connection().cursor()
        for transaction_id, predicted_category_id, probabilities in df_filtered[["transaction_id", "predicted_category_id", "probability"]].values:
            cursor.execute("UPDATE transactions SET category_key = ? WHERE rowid = ?", (int(predicted_category_id), int(transaction_id)))
        db.get_connection().commit()
    console.log(f"{len(df)} transactions processed.",style="green")
    console.log(f"{len(df_filtered)} transactions categorized.",style="green")


@cli.command()
# fmt: off
@click.option("--config-profile",help="Configuration profile to use to read config values from instead of specifying on the command line", default="default", show_default=True, required=False)
@click.option("--db-password", help="Encryption password of moneymoney DB")
@click.option("--date-from", type=RelativeDate(), default="2Y", help="Oldest transaction to use for training (e.g., -1Y for one year ago or 2021-01-01 for an absolute date)", required=True, show_default=True)  # noqa: E501
@click.option("--date-to", type=RelativeDate(), default=datetime.datetime.now().strftime("%Y-%m-%d"), help="Newest transaction to use for training (e.g., -3M for three months ago or 2021-01-01 for an absolute date)", required=True, show_default=True)  # noqa: E501
@click.option("--limit-to-account", help="Limit training to transactions in the defined account. Can be provided multiple times",multiple=True)
@click.option("--model-name", help="Specify the model name to be created", required=True, default="default", show_default=True)
@click.option("--evaluate", help="Evaluate the model by splitting the data into training and testset and showing a report", is_flag=True)
@click.option("--test-set-size", help="Size of the test set in percent", default=0.2, show_default=True)
# fmt: on
def train_model(config_profile,db_password, date_from:datetime.datetime, date_to, limit_to_account, model_name,evaluate, test_set_size):
    config = read_config(config_profile)
    db_password = db_password if db_password is not None else config.get("db_password")
    limit_to_account = limit_to_account if limit_to_account else config.get("limit_to_accounts")
    model_name = model_name if model_name is not None else config.get("model_name")
    date_from = date_from if date_from is not None else config.get("date_from")
    date_to = date_to if date_to is not None else config.get("date_to")
    if None in [db_password, model_name, date_from, date_to]:
        raise click.UsageError("Please provide all required parameters")
    try:
        db = MoneyMoneyDB(db_password)
    except MoneyMoneyDB.Exception:
        console.log("Error opening database", style="red")
        return

    # Prepare dataframe
    console.log(f"Reading transactions from database from {date_from.strftime("%Y-%m-%d")} to {date_to.strftime("%Y-%m-%d")}...")
    df = pd.DataFrame(db.get_transactions(date_from, date_to, limit_to_account))
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["purpose"] = df["purpose"].fillna("no purpose")
    df["name"] = df["name"].fillna("no name")
    df["type"] = df["type"].fillna("no type")

    if "limit_to_categories" in config:
        limit_to_category = config.get("limit_to_categories")
        df = df[df["category_key"].isin(limit_to_category)]

    X = df[["amount", "purpose", "name", "type"]]
    y = df["category_key"]
    scaler = StandardScaler()

    # Use Word2Vec for text fields
    word2vec_vectorizer = Word2VecVectorizer()

    # Split into training and testing sets
    if evaluate:
        console.log("Splitting data into training and testing sets...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_set_size, random_state=42)
    else:
        X_train, y_train = X, y
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
    console.log("Training model...")
    model.fit(X_train, y_train)

    # Evaluate the model
    if evaluate:
        console.log("Evaluating model...")
        y_pred = model.predict(X_test)

        # Calculate classification report
        actual_classes = sorted(set(y_test))
        categories = db.get_categories()
        actual_class_names = [categories[cls] for cls in actual_classes]

        print(classification_report(y_test, y_pred, labels=actual_classes, target_names=actual_class_names))
        console.log ("Model was not saved. Re-run without --evaluate to save the model",style="yellow")
    else:
        # Save the model
        model_file_name = db.get_data_dir().joinpath(model_name+ ".pkl") 
        console.log(f"Saving model {model_file_name}")
        joblib.dump(model, model_file_name)
        # save model meta data
        model_meta = {
            "model_name": model_name,
            "date_from": date_from,
            "date_to": date_to,
            "limit_to_account": limit_to_account,
            "limit_to_category": limit_to_category,
            "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(model_file_name.with_suffix(".yml"), "w") as file:
            yaml.dump(model_meta, file)


if __name__ == "__main__":
    cli(obj={})
