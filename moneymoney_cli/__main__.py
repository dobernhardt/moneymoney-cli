import datetime
import shutil

import click
import getch
import joblib
import numpy as np
import pandas as pd
import yaml
from rich.table import Table
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import read_config
from .console import console
from .moneymoney import MoneyMoney
from .relative_date import RelativeDate
from .version import __version__
from .word_vec import Word2VecVectorizer


@click.group()
@click.version_option(__version__, prog_name="MoneyMoney CLI")
def cli():
    pass


@cli.command()
def list_accounts():
    mm = MoneyMoney()
    table = Table(title="Accounts")

    table.add_column("UUID", justify="left", style="cyan", no_wrap=True)
    table.add_column("Account name", justify="left", style="magenta")

    for account in mm.get_accounts():
        table.add_row(account["uuid"], account["name"])

    console.print(table)


@cli.command()
def list_categories():
    mm = MoneyMoney()
    table = Table(title="Categories")

    table.add_column("UUID", justify="left", style="cyan", no_wrap=True)
    table.add_column("Account name", justify="left", style="magenta")

    for account in mm.get_categories():
        table.add_row(account["uuid"], account["name"])

    console.print(table)


@cli.command()
# fmt: off
@click.option("--config-profile", help="Configuration profile to use to read config values from instead of specifying on the command line", default="default", show_default=True, required=False)  # noqa: E501
@click.option("--date-from", type=RelativeDate(), default="2Y", help="Oldest transaction to be categorized (e.g., 1Y for one year ago or 2021-01-01 for an absolute date)", required=True, show_default=True)  # noqa: E501
@click.option("--date-to", type=RelativeDate(), default=(datetime.datetime.now() + datetime.timedelta(days=30)).strftime("%Y-%m-%d"), help="Newest transaction to be categorized", required=True, show_default=True)  # noqa: E501
@click.option("--limit-to-account", help="Limit classification to transactions in the defined account. Can be provided multiple times", multiple=True)
# fmt: on
def list_category_usage(config_profile, date_from, date_to, limit_to_account):
    """
    List the usage of categories in the transactions.

    Parameters:
    - date_from: Oldest transaction to be categorized (e.g., 1Y for one year ago or 2021-01-01 for an absolute date).
    - date_to: Newest transaction to be categorized.
    - limit_to_account: Limit classification to transactions in the defined account. Can be provided multiple times.

    This function lists the usage of categories in the transactions within the specified date range and accounts.
    """
    config = read_config(config_profile)
    limit_to_account = limit_to_account if limit_to_account else config.get("limit_to_accounts")
    mm = MoneyMoney()
    category_usage = mm.get_category_usage(date_from, date_to, limit_to_account)
    # Sort category_usage by the number of transactions
    sorted_category_usage = sorted(category_usage.items(), key=lambda item: item[1], reverse=True)

    table = Table(title="Category Usage")

    table.add_column("Category ID", justify="left", style="cyan", no_wrap=True)
    table.add_column("Category name", justify="left", style="magenta")
    table.add_column("# of transactions", justify="left", style="green")

    for category_id, usage_count in sorted_category_usage:
        table.add_row(str(category_id), mm.get_category_name(category_id), str(usage_count))

    console.print(table)


@cli.command()
# fmt: off
@click.option("--config-profile", help="Configuration profile to use to read config values from instead of specifying on the command line", default="default", show_default=True, required=False)  # noqa: E501
@click.option("--date-from", type=RelativeDate(), default="3M", help="Oldest transaction to be categorized (e.g., 1Y for one year ago or 2021-01-01 for an absolute date)", required=True, show_default=True)  # noqa: E501
@click.option("--date-to", type=RelativeDate(), default=(datetime.datetime.now() + datetime.timedelta(days=30)).strftime("%Y-%m-%d"), help="Newest transaction to be categorized", required=True, show_default=True)  # noqa: E501
@click.option("--limit-to-account", help="Limit classification to transactions in the defined account. Can be provided multiple times", multiple=True)
@click.option("--model-name", help="Specify the model to be used", default="default", show_default=True)
@click.option("--overwrite", help="Categorize also already categorized transactions", is_flag=True)
@click.option("--list-only", help="List only the transactions how the would be categorized. Does not alter MoneyMoney", is_flag=True)
@click.option("--probability-threshold", help="Probability threshold to auto apply the category without asking for user input", default=0.9, show_default=True)
@click.option("--auto-apply", help="Autoapply if propability>propability threshold without user input", is_flag=True)
# fmt: on
def categorize(date_from, date_to, limit_to_account, model_name, overwrite, config_profile, list_only, probability_threshold, auto_apply):
    """
    Categorize transactions using a trained machine learning model.
    Parameters can either be provided as command line arguments or read from a configuration file. If a config.yml file is either found in the current directory
    or in the MoneyMoney data directory it will be read and configuraiton from the specified profile will be used.
    Any parameters provided as command line arguments will override the configuration file values.
    All transactions withing the specified date range and accounts will be categorized using the specified model.
    The categorization can interactively be applied to MoneyMoney.

    """
    config = read_config(config_profile)
    limit_to_account = limit_to_account if limit_to_account else config.get("limit_to_accounts")
    model_name = model_name if model_name is not None else config.get("model_name")
    date_from = date_from if date_from is not None else config.get("date_from")
    date_to = date_to if date_to is not None else config.get("date_to")
    mm = MoneyMoney()
    console.log("Reading transactions from database...")
    transactions = list(mm.get_transactions(date_from, date_to, limit_to_account, only_uncategorized=(not overwrite)))

    # Create a DataFrame from the transactions
    df = pd.DataFrame([t.__dict__ for t in transactions])
    if df.empty:
        console.log("No transactions to categorize.", style="red")
        return

    # Calculate the number of transactions
    num_transactions = len(transactions)

    # Extract the accounts
    accounts = set(transaction.account_uid for transaction in transactions)
    account_names = [mm.get_account_name(account) for account in accounts]

    # Find the oldest and newest dates of transactions
    oldest_date = min(transaction.timestamp for transaction in transactions)
    newest_date = max(transaction.timestamp for transaction in transactions)

    console.log(f"Number of transactions found: {num_transactions}", style="green")
    console.log(f"Accounts for which transactions have been found: {', '.join(account_names)}", style="green")
    console.log(f"Oldest transaction date: {datetime.datetime.fromtimestamp(oldest_date).strftime('%Y-%m-%d')}", style="green")
    console.log(f"Newest transaction date: {datetime.datetime.fromtimestamp(newest_date).strftime('%Y-%m-%d')}", style="green")

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["purpose"] = df["purpose"].fillna("no purpose")
    df["name"] = df["name"].fillna("no name").str.replace("VISA", "", regex=False)
    X = df[["amount", "purpose", "name"]]

    console.log(f"Loading model {model_name}...")
    model = joblib.load(str(mm.get_data_dir().joinpath(model_name)) + ".pkl")

    # Predict categories and probabilities
    console.log("Predicting categories...")
    predicted_probabilities = model.predict_proba(X)

    # Get the top 3 categories for each transaction
    top_3_indices = np.argsort(predicted_probabilities, axis=1)[:, -3:][:, ::-1]
    top_3_probabilities = np.sort(predicted_probabilities, axis=1)[:, -3:][:, ::-1]

    # Assuming the model returns UUIDs directly
    top_3_categories = [[model.classes_[index] for index in indices] for indices in top_3_indices]

    df["top_3_category_uuids"] = top_3_categories
    df["top_3_categories"] = [[mm.get_category_name(category) for category in categories] for categories in top_3_categories]
    df["top_3_probabilities"] = [probabilities for probabilities in top_3_probabilities]

    if list_only:
        table = Table()
        table.add_column("Date", justify="left")
        table.add_column("Purpose", justify="left")
        table.add_column("Name", justify="left")
        table.add_column("Amount", justify="right")
        table.add_column("Category", justify="left")
        table.add_column("Propability", justify="left")
        for _, row in df.iterrows():
            probability = row["top_3_probabilities"][0]
            category = row["top_3_categories"][0]
            probability_style = "bright_black" if probability < probability_threshold else "white"
            row = table.add_row(
                datetime.datetime.fromtimestamp(row["timestamp"]).strftime("%Y-%m-%d"),
                row["purpose"],
                row["name"],
                f"{row['amount']:.2f}",
                category,
                f"{probability:.2f}",
                style=probability_style,
            )
        console.print(table)
        return
    console.log("List/Apply categorization to MoneyMoney")
    console.log("Press 1, 2, 3 to select category or ESC to break or Enter to skip")
    # Define column widths as percentages
    column_widths = {"Date": 10, "Purpose": 20, "Name": 20, "Amount": 10, "Probable Categories": 40}
    terminal_width = shutil.get_terminal_size().columns

    # Calculate column widths in characters
    column_widths_chars = {key: int(terminal_width * (value / 100)) for key, value in column_widths.items()}

    transaction_id = 0
    num_categorized = 0
    num_skipped = 0
    for _, row in df.iterrows():
        console.print("\n\n")
        console.print(f"Transaction {transaction_id + 1} of {num_transactions}", style="green")
        transaction_id += 1
        table = Table()
        table.add_column("Date", justify="left", style="cyan", no_wrap=True, width=column_widths_chars["Date"])
        table.add_column("Purpose", justify="left", style="magenta", width=column_widths_chars["Purpose"])
        table.add_column("Name", justify="left", style="yellow", width=column_widths_chars["Name"])
        table.add_column("Amount", justify="right", style="green", width=column_widths_chars["Amount"])
        table.add_column("Probable Categories", justify="left", style="blue", width=column_widths_chars["Probable Categories"])
        table.add_row(
            datetime.datetime.fromtimestamp(row["timestamp"]).strftime("%Y-%m-%d"),
            row["purpose"],
            row["name"],
            f"{row['amount']:.2f}",
            "\n".join(
                [
                    f"{i + 1}. {category} (Probability: {probability:.2f})"
                    for i, (category, probability) in enumerate(zip(row["top_3_categories"], row["top_3_probabilities"]))
                ]
            ),
        )
        console.print(table)
        if auto_apply and row["top_3_probabilities"][0] > probability_threshold:
            console.log("Transaction automatically categorized as: ", row["top_3_categories"][0], style="green")
            selected_category = row["top_3_category_uuids"][0]
        else:
            choice = getch.getch()
            while choice not in ["1", "2", "3", "\x1b", "\n"]:
                console.print("Invalid choice. Select category (1, 2, 3) or ESC for break or Enter for skip: ")
                choice = getch.getch()
            if choice == "\x1b":
                num_skipped += 1
                break
            if choice == "\n":
                console.log("Skipping transaction", style="yellow")
                num_skipped += 1
                continue
            console.log("Transaction categorized as: ", row["top_3_categories"][int(choice) - 1], style="green")
            selected_category = row["top_3_category_uuids"][int(choice) - 1]
        num_categorized += 1
        mm.categorize_transaction(row["transaction_id"], selected_category)

    console.log(f"{transaction_id} transactions processed.", style="green")
    console.log(f"{num_categorized} transactions categorized.", style="green")
    console.log(f"{num_skipped} transactions skipped.", style="green")


@cli.command()
# fmt: off
@click.option("--config-profile", help="Configuration profile to use to read config values from instead of specifying on the command line", default="default", show_default=True, required=False)  # noqa: E501
@click.option("--date-from", type=RelativeDate(), default="2Y", help="Oldest transaction to use for training (e.g., -1Y for one year ago or 2021-01-01 for an absolute date)", required=True, show_default=True)  # noqa: E501
@click.option("--date-to", type=RelativeDate(), default=datetime.datetime.now().strftime("%Y-%m-%d"), help="Newest transaction to use for training (e.g., -3M for three months ago or 2021-01-01 for an absolute date)", required=True, show_default=True)  # noqa: E501
@click.option("--limit-to-account", help="Limit training to transactions in the defined account. Can be provided multiple times", multiple=True)
@click.option("--model-name", help="Specify the model name to be created", required=True, default="default", show_default=True)
@click.option("--evaluate", help="Evaluate the model by splitting the data into training and testset and showing a report", is_flag=True)
@click.option("--test-set-size", help="Size of the test set in percent", default=0.2, show_default=True)
# fmt: on
def train_model(config_profile, date_from: datetime.datetime, date_to, limit_to_account, model_name, evaluate, test_set_size):
    config = read_config(config_profile)
    limit_to_account = limit_to_account if limit_to_account else config.get("limit_to_accounts")
    model_name = model_name if model_name is not None else config.get("model_name")
    date_from = date_from if date_from is not None else config.get("date_from")
    date_to = date_to if date_to is not None else config.get("date_to")
    if None in [model_name, date_from, date_to]:
        raise click.UsageError("Please provide all required parameters")
    moneymoney = MoneyMoney()

    # Prepare dataframe
    console.log(f"Reading transactions from database from {date_from.strftime('%Y-%m-%d')} to {date_to.strftime('%Y-%m-%d')}...")
    df = pd.DataFrame(moneymoney.get_transactions(date_from, date_to, limit_to_account))
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["purpose"] = df["purpose"].fillna("no purpose")
    df["name"] = df["name"].fillna("no name").str.replace("VISA", "", regex=False)

    if "limit_to_categories" in config:
        limit_to_category = config.get("limit_to_categories")
        df = df[df["category_uid"].isin(limit_to_category)]
    else:
        limit_to_category = []

    console.log("Assessing training data...")
    # Print table with category usage
    category_counts = df["category_uid"].value_counts()
    table = Table(title="Transactions used per category for training")
    table.add_column("Category ID", justify="left", style="cyan", no_wrap=True)
    table.add_column("Category Name", justify="left", style="magenta")
    table.add_column("# of Transactions", justify="right", style="green")

    for category_id, count in category_counts.items():
        table.add_row(category_id, moneymoney.get_category_name(category_id), str(count))
    console.print(table)

    X = df[["amount", "purpose", "name"]]
    y = df["category_uid"]
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
        actual_class_names = [moneymoney.get_category_name(cls) for cls in actual_classes]

        print(classification_report(y_test, y_pred, labels=actual_classes, target_names=actual_class_names))
        console.log("Model was not saved. Re-run without --evaluate to save the model", style="yellow")
    else:
        # Save the model
        model_file_name = moneymoney.get_data_dir().joinpath(model_name + ".pkl")
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


@cli.command()
# fmt: off
@click.option("--config-profile", help="Configuration profile to use to read config values from instead of specifying on the command line", default="default", show_default=True, required=False)  # noqa: E501
@click.option("--limit-to-account", help="Limit training to transactions in the defined account. Can be provided multiple times", multiple=True)
@click.option("--date-from", type=RelativeDate(), default="2Y", help="Oldest transaction to use for training (e.g., -1Y for one year ago or 2021-01-01 for an absolute date)", required=True, show_default=True)  # noqa: E501
@click.option("--date-to", type=RelativeDate(), default=datetime.datetime.now().strftime("%Y-%m-%d"), help="Newest transaction to use for training (e.g., -3M for three months ago or 2021-01-01 for an absolute date)", required=True, show_default=True)  # noqa: E501
def list_transactions(config_profile, date_from, date_to, limit_to_account):
    config = read_config(config_profile)
    date_from = date_from if date_from is not None else config.get("date_from")
    date_to = date_to if date_to is not None else config.get("date_to")
    limit_to_account = limit_to_account if limit_to_account else config.get("limit_to_accounts")
    mm = MoneyMoney()
    console.log(f"Reading transactions from database from {date_from.strftime('%Y-%m-%d')} to {date_to.strftime('%Y-%m-%d')}...")

    table = Table(title="Transactions")
    table.add_column("Date", justify="left", style="cyan", no_wrap=True)
    table.add_column("Amount", justify="right", style="magenta")
    table.add_column("Purpose", justify="left", style="green")
    table.add_column("Name", justify="left", style="yellow")
    table.add_column("Category", justify="left", style="blue")

    for transaction in mm.get_transactions(date_from, date_to, limit_to_account):
        table.add_row(
            datetime.datetime.fromtimestamp(transaction.timestamp).strftime("%Y-%m-%d"),
            f"{transaction.amount:.2f}",
            transaction.purpose,
            transaction.name,
            mm.get_category_name(transaction.category_uid),
        )

    console.print(table)


if __name__ == "__main__":
    cli(obj={})
