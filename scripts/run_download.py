"""Download URLs listed in config/sources.json."""

import argparse
from siniestros_viales.download_data import download_configured


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Replace existing files")
    args = parser.parse_args()
    download_configured(force=args.force)


if __name__ == "__main__":
    main()
