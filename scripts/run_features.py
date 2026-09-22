"""Assemble clean years and save an enriched modeling dataset."""

import argparse

from siniestros_viales.build_dataset import build_dataset
from siniestros_viales.config import INTERIM, PROCESSED
from siniestros_viales.features import add_features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="+", type=int, default=list(range(2018, 2025)))
    args = parser.parse_args()
    paths = [INTERIM / f"atus_{year}.parquet" for year in args.years]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        parser.error(f"Missing cleaned files: {missing}")
    result = add_features(build_dataset(paths))
    result.to_parquet(PROCESSED / "modeling_dataset.parquet", index=False)


if __name__ == "__main__":
    main()
