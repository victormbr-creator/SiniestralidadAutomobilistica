"""Fit the three models using the fixed temporal split."""

import argparse
import pandas as pd

from siniestros_viales.config import PROCESSED
from siniestros_viales.train_models import train_models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=["victimas", "fatal"], default="victimas")
    parser.add_argument("--class-weight", choices=["balanced", "none"], default="balanced")
    args = parser.parse_args()
    path = PROCESSED / "modeling_dataset.parquet"
    if not path.exists():
        parser.error(f"Missing dataset: {path}; run make features first")
    train_models(pd.read_parquet(path), target=args.target, class_weight=None if args.class_weight == "none" else args.class_weight)


if __name__ == "__main__":
    main()
