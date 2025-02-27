import yaml
import pathlib
from .console import console
from schema import Optional, Schema, SchemaError


config_schema = Schema(
    {
        "profiles": [
            {
                "profile_name": str,
                Optional("model_name"): str,
                Optional("date_from"): str,
                Optional("date_to"): str,
                Optional("limit_to_accounts"): [str],
                Optional("limit_to_categories"): [str],
            }
        ],
    }
)


def read_config(profile: str = None):
    configfile = pathlib.Path(MoneyMoneyDB.get_data_dir().joinpath("moneymoney-cli.config"))
    if not configfile.exists():
        configfile = pathlib.Path("moneymoney-cli.config")
    if not configfile.exists():
        console.log("No config file found", style="yellow")
        return {}
    console.log(f"Reading config from {configfile.absolute()}")
    with open(configfile, "r") as file:
        config = yaml.safe_load(file)
        try:
            config_schema.validate(config)
        except SchemaError as e:
            console.log(f"Config file is invalid: {e}.", style="red")
            console.log("Ignoring config file", style="red")
            return {}
        if profile is not None:
            for prof in config.get("profiles", []):
                if prof.get("profile_name") == profile:
                    config.update(prof)
                    break
        del config["profiles"]
    return config
