"""Clean one or more confirmed ATUS files: YEAR=PATH pairs."""

import argparse
from pathlib import Path

from siniestros_viales.clean_atus import clean_files, load_mapping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", metavar="YEAR=PATH", help="Example: 2018=data/raw/atus_2018.csv")
    args = parser.parse_args()
    if not args.files:
        parser.error("Supply YEAR=PATH pairs after confirming the source schema and mapping")
    files = {}
    for item in args.files:
        year, separator, path = item.partition("=")
        if not separator or not year.isdigit() or not Path(path).is_file():
            parser.error(f"Invalid YEAR=PATH: {item}")
        files[int(year)] = Path(path)
    clean_files(files, load_mapping())


if __name__ == "__main__":
    main()
