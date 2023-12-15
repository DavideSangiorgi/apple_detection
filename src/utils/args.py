import argparse
from pathlib import Path

###

DEFAULT_CONFIG_PATH = Path().joinpath("configs", "default.json")

###


def parse_args():
    parser = argparse.ArgumentParser(description="Apple detection.")

    parser.add_argument(
        "--config",
        required=False,
        default=str(DEFAULT_CONFIG_PATH),
        type=str,
        help="configuration file path",
    )

    #

    args = parser.parse_args()

    #

    return args
