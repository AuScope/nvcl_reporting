#!/usr/bin/env python3
"""
Downloads the NVCL store CSV from https://nvclstore.data.auscope.org.au/all.csv
and saves it to a local file.
"""
import argparse
import logging
import sys
from pathlib import Path

import requests

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


NVCL_ALL_CSV_URL = 'https://nvclstore.data.auscope.org.au/all.csv'
DEFAULT_OUTPUT = 'metadata.csv'


def download_csv(url: str | None, output_path: Path) -> None:
    """
    Download a CSV file from a URL and write it to disk.

    :param url: URL of the CSV file to download, if None uses default NVCL cache 'all.csv' URL
    :param output_path: local path to write the downloaded file
    """
    if url is None:
        url = NVCL_ALL_CSV_URL
    logger.info("Downloading %s", url)
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error("Failed to download %s: %s", url, e)
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(response.content)
    logger.info("Saved %d bytes to %s", len(response.content), output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='nvcl_store',
        description='Downloads the NVCL store CSV from AuScope',
    )
    parser.add_argument(
        '-o', '--output',
        default=DEFAULT_OUTPUT,
        help=f'output file path (default: {DEFAULT_OUTPUT})',
    )
    parser.add_argument(
        '-u', '--url',
        default=NVCL_ALL_CSV_URL,
        help=f'URL to download from (default: {NVCL_ALL_CSV_URL})',
    )

    args = parser.parse_args()
    download_csv(args.url, Path(args.output))
