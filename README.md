
# MoneyMoney CLI Tool

MoneyMoney CLI Tool is a command-line tool for listing and manipulating the MoneyMoney database. The main feature is probably categorizing transactions in the MoneyMoney application using machine learning models.

So far this is a very early development version - use it on your own risk

## Features

- List accounts in the MoneyMoney database
- Categorize transactions using a trained machine learning model
- Apply categorization to the database



## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/moneymoney-classifier.git
    cd moneymoney-classifier
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv .venv
    source .venv/bin/activate
    ```

3. Locally install the package and dependencies:
    ```sh
    pip install -e .
    ```

## Usage

### List Accounts

To list all accounts in the MoneyMoney database, use the `list_accounts` command:

```sh
python -m moneymoney_cli list_accounts --db-password YOUR_DB_PASSWORD
```

### Categorize Transactions

To categorize transactions using a trained model, use the `categorize` command:

```sh
python -m moneymoney_cli categorize --db-password YOUR_DB_PASSWORD --date-from -3M --date-to 2023-01-01 --limit-to-account ACCOUNT_ID --model-name default --apply
```

- `--db-password`: Encryption password of the MoneyMoney database.
- `--date-from`: Oldest transaction to be categorized (e.g., `-1Y` for one year ago or `2021-01-01` for an absolute date).
- `--date-to`: Newest transaction to be categorized.
- `--limit-to-account`: Limit classification to transactions in the defined account. Can be provided multiple times.
- `--model-name`: Specify the model to be used.
- `--apply`: Apply the categorization to the database.

### Train Model
To train a new model, use the train_model command:

- `--db-password`: Encryption password of the MoneyMoney database.
- `--date-from`: Oldest transaction to use for training (e.g., -1Y for one year ago or 2021-01-01 for an absolute date).
- `--date-to`: Newest transaction to use for training.
- `--limit-to-account`: Limit training to transactions in the defined account. Can be provided multiple times.
- `--model-name`: Specify the model name to be created.
- `--limit-to-category-file`: Limit the training to category IDs specified in the provided text file. It is recommended to use this option to limit the training to those categories that have enough sample data. The tool expects a category ID each line followed by a whitespace or linebreak. Anything after the whitespace is ignored.


## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Acknowledgements

- [MoneyMoney](https://moneymoney-app.com/) - The personal finance software used in this project.
- [Scikit-learn](https://scikit-learn.org/) - The machine learning library used for training and prediction.
```

