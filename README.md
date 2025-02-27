
# MoneyMoney CLI Tool

MoneyMoney CLI Tool is a command-line tool for listing and manipulating MoneyMoney data by using the AppleScript API. The main feature is probably categorizing transactions in the MoneyMoney application using machine learning models.

So far this is a very early development version - use it on your own risk

## Features

- List accounts in the MoneyMoney database
- List category usage - i.e. sort categories by the numer of transactions they are used for
- Categorize transactions using a trained machine learning model
- Apply categorization to the database

Most configuraiton parameters can also be provided in a config file instead of the commandline


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



## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Acknowledgements

- [MoneyMoney](https://moneymoney-app.com/) - The personal finance software used in this project.
- [Scikit-learn](https://scikit-learn.org/) - The machine learning library used for training and prediction.
```

