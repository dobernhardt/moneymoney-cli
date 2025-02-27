import pathlib
import plistlib
import subprocess
import sys
from dataclasses import dataclass

from .console import console


class MoneyMoney:

    class Exception(Exception):
        ...

    def __init__(self):
        try:
            output = self.run_script('tell application "MoneyMoney" to get version')
            version = output.decode().strip()
            console.log(f"MoneyMoney is running version {version}", style="green")
            self._default_category = self.get_default_category()
            self._categorynames = {category['uuid']: category['name'] for category in self.get_categories()}
            self._accountnames = {account['uuid']: account['name'] for account in self.get_accounts()}
        except MoneyMoney.Exception as e:
            console.log("Unable to access MoneyMoney", style="red")
            console.log (e, style="red")
            sys.exit(1)

    @staticmethod
    def get_data_dir() -> pathlib.Path:
        return pathlib.Path.home().joinpath("Library/Containers/com.moneymoney-app.retail/Data/Library/Application Support/MoneyMoney/Database").absolute()


    def get_account_name(self, account_uuid):
        return self._accountnames[account_uuid]

    def get_category_name(self, category_uuid):
        return self._categorynames[category_uuid]

    def run_script(self, script):
        script = f"""
        tell application "MoneyMoney"
            {script}
        end tell"""
        command = ['osascript', '-e', script]
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as pipe:
            result = pipe.communicate()
            if result[1] or pipe.returncode != 0:
                raise MoneyMoney.Exception('Failed to execute MoneyMoney command. %s' % result[1].decode().strip())
        return result[0]
                

    def get_default_category(self):
        script = "export categories"
        plist_output = self.run_script(script)
        # Parse the plist
        categories_plist = plistlib.loads(plist_output)
        for category in categories_plist:
            if category['default']:
                return {'uuid': category['uuid'], 'name': category['name']}
            

    def categorize_transaction(self, transaction_id, category_uuid):
        script = f"set transaction id {transaction_id} category to \"{category_uuid}\""
        self.run_script(script)

    def get_categories(self):
        script = "export categories"
        plist_output = self.run_script(script)
        # Parse the plist
        categories_plist = plistlib.loads(plist_output)
        # Extract UUID and Name
        stack = []
        for category in categories_plist:
            # Maintain stack structure
            while stack and stack[-1][1] >= category['indentation']:
                stack.pop()
            # Append the current element
            if stack:
                full_name = f"{'/'.join(x[0] for x in stack)}/{category['name']}"
            else:
                full_name = category['name']
            # Push to stack
            stack.append((category['name'], category['indentation']))
            # Store if it's a leaf
            if not category['group']:
                yield {'uuid': category['uuid'], 'name': full_name}

    def get_category_usage(self, date_from, date_to, limit_to_accounts=None):
        transactions = self.get_transactions(date_from, date_to, limit_to_accounts)
        category_usage = {}
        for transaction in transactions:
            category = transaction.category_uid
            if category not in category_usage:
                category_usage[category] = 0
            category_usage[category] += 1
        return category_usage


    def get_accounts(self):
        script = "export accounts"
        plist_output = self.run_script(script)
        accounts_plist = plistlib.loads(plist_output)
        stack = []
        for account in accounts_plist:
            # Maintain stack structure
            while stack and stack[-1][1] >= account['indentation']:
                stack.pop()
            # Append the current element
            if stack:
                full_name = f"{'/'.join(x[0] for x in stack)}/{account['name']}"
            else:
                full_name = account['name']
            # Push to stack
            stack.append((account['name'], account['indentation']))
            # Store if it's a leaf
            if not account['group']:
                yield {'uuid': account['uuid'], 'name': full_name}
                
            

    
    @dataclass
    class Transaction:
        transaction_id: int
        account_uid: str
        amount: float
        purpose: str
        name: str
        category_uid: str
        timestamp: int
    

    def get_transactions(self, date_from, date_to, limit_to_accounts=None, only_uncategorized=False):
        script = f'export transactions from date "{date_from.strftime("%Y-%m-%d")}" to date "{date_to.strftime("%Y-%m-%d")}" as "plist"'
        script_output = self.run_script(script)
        # Convert the string output to bytes
        transactions_plist = plistlib.loads(script_output)
        for transaction in transactions_plist['transactions']:
            if limit_to_accounts and transaction['accountUuid'] not in limit_to_accounts:
                continue
            if only_uncategorized and transaction['categoryUuid'] != self._default_category["uuid"]:
                continue
            yield MoneyMoney.Transaction(
                transaction_id=transaction['id'],
                account_uid=transaction['accountUuid'],
                amount=transaction['amount'],
                purpose=transaction.get('purpose', ''),
                name=transaction['name'],
                category_uid=transaction['categoryUuid'],
                timestamp=transaction['bookingDate'].timestamp()
            )


