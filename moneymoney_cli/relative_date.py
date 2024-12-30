import click
import datetime


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
