"""Download only explicitly configured source files."""

from pathlib import Path
from urllib.parse import urlparse

import requests

from .config import RAW, SOURCES
from .utils import get_logger, read_json

LOG = get_logger(__name__)


def download_file(url: str, destination: Path, force: bool = False, timeout: int = 60) -> Path:
    """Stream an HTTPS file to disk; skip existing files unless forced."""
    if urlparse(url).scheme != "https":
        raise ValueError("Only HTTPS source URLs are accepted")
    if destination.exists() and not force:
        LOG.info("Skipping existing file: %s", destination)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    try:
        with requests.get(url, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    LOG.info("Downloaded %s", destination)
    return destination


def download_configured(config_path: Path = SOURCES, force: bool = False) -> list[Path]:
    """Download entries with ``url`` and relative ``filename`` in sources.json."""
    entries = read_json(config_path).get("sources", [])
    if not entries:
        LOG.warning("No source URLs configured in %s", config_path)
    results = []
    for entry in entries:
        filename = Path(entry["filename"])
        if filename.is_absolute() or ".." in filename.parts:
            raise ValueError("Source filename must stay within data/raw")
        results.append(download_file(entry["url"], RAW / filename, force=force))
    return results
