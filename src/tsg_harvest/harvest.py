#!/usr/bin/env python3
import os
from zipfile import ZipFile
import datetime
import configparser
import csv
import sys
from pathlib import Path, PurePath, PureWindowsPath
import argparse


import threddsclient
import requests


# Directory where TSG files will be written to
DOWNLOAD_DIR = "/datasets/mr-geochem/work/nci_ncvl_collection"

# URL of NVCL TSG datasets
THREDDS_CAT = 'https://thredds.nci.org.au/thredds/catalog/rs07/{prov}/catalog.xml'

# NVCLDataServices providers
PROVIDERS = ['NSW', 'Vic', 'Tas', 'Qld', 'CSIRO', 'SA', 'NT', 'WA']

HL_SCAN_DATE = 'hl scan date' # Date of core scan by Hylogger
TSG_PUBLISH_DATE = 'tsg publish date' # Last modified date of TSG file in THREDDS filesystem
NVCL_ID = 'drillhole name'

# CSV fields extracted from a TSG file
TSG_FIELDS = [HL_SCAN_DATE, 'domain id', 'imagelog id', 'proflog id', 'traylog id', 'seclog id', NVCL_ID, 'dataset name', 'lat lon', 'collar']


def parse_date(date_str: str) -> datetime:
    """
    Parse scan date string take from TSG file

    :param date_str: date string
    :returns: datetime object, if string can't be parsed returns a datetime of the year 9999
    """
    try:
        dt = datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
    except ValueError as ve:
        print(f"Cannot parse scan date '{date_str}': {ve}")
        return datetime.datetime.max
    return dt


def get_tsg_metadata(filepath: Path) -> dict:
    """ Reads metadata from TSG file

    :param filename: filename of TSG file
    :returns: datetime or datetime.max upon error
    """
    print(f"Finding metadata in {filepath}")
    config = configparser.ConfigParser(allow_no_value=True, strict=False)
    try:
        # Try utf-8
        config.read(str(filepath))
    except UnicodeDecodeError:
        # Try cp1252
        print(f"Cannot read {filepath} using 'utf-8' coding, now trying 'cp1252'")
        config.read(str(filepath), encoding='cp1252')

    # Look for scan date
    try:
        date_str = config['dbextra']['scan date']
    except (configparser.Error, KeyError):
        print(f"WARN - {filepath} does not have scan date")
        # Look for create date
        date_str = config['description']['Created']
        print(f"Using created date {date_str} instead")

    # Parse scan date
    print(f"Found scan date {date_str}")
    field_dict =  {HL_SCAN_DATE: parse_date(date_str) }

    # Look for fields in TSG file
    for field in ['domain id', 'imagelog id', 'proflog id', 'traylog id', 'seclog id', 'drillhole name', 'dataset name', 'lat lon', 'collar']:
        try:
            field_dict[field] = config['dbextra'][field]
        except (configparser.Error, KeyError):
            print(f"Error in TSG file: {filepath} does not have {field}")
            field_dict[field] = ''
    return field_dict

def process_prov(prov: str, csvwriter):
    """
    Extract data for one provider

    :param prov: provider name - see 'PROVIDERS' list
    :param csvwrite: instance of csvwriter class to output each data point
    """
    cat = threddsclient.read_url(THREDDS_CAT.format(prov=prov))
    prov_dir = os.path.join(DOWNLOAD_DIR, prov)
    if not os.path.exists(prov_dir):
        os.mkdir(prov_dir)
    for ds in cat.flat_datasets():
        # Fetch THREDDS modified date for TSG file
        try:
            tsg_mod_date = datetime.datetime.strptime(ds.modified, "%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            tsg_mod_date = datetime.datetime.max
        print(f"Found THREDDS modified date {tsg_mod_date}")

        # Has TSG ZIP file been downloaded?
        zip_file = os.path.join(prov_dir, ds.name)
        if not(os.path.exists(zip_file)):
            print(f"Downloading {ds.download_url()=}")
            # Download TSG ZIP file
            r = requests.get(ds.download_url())  
            with open(zip_file, 'wb') as fd:
                fd.write(r.content)

        # Unzip TSG files
        with ZipFile(zip_file) as z:
            unzip_paths = []
            # We don't know if this is a POSIX path or a Windows path
            # So this will convert either kind to a POSIX path
            for p in z.namelist():
                pwp = PureWindowsPath(p)
                pp = PurePath(pwp.as_posix())
                unzip_paths.append(pp) 

            # Check if zip file has been unzipped, looking for TSG the file
            path = Path(prov_dir)
            tsg_path = [uzp for uzp in unzip_paths if str(uzp)[-8:] == '_tsg.tsg']
            if len(tsg_path) > 0 and not path.joinpath(tsg_path[0]).exists():
                print(f"Unzipping {zip_file}")
                # This forces 'Zipfile' to create POSIX paths using the Windows paths in the ZIP file
                os.path.altsep = '\\'
                z.extractall(path=prov_dir)

            # Continue to next loop if there is no TSG file
            if not path.joinpath(tsg_path[0]).exists():
                print("ERROR - Cannot find TSG file in {zip_file}")
                continue

            # Extract scan date and other metadata from TSG file
            for unz_path in unzip_paths:
                # Look for TSG metadata file
                if str(unz_path)[-8:] == '_tsg.tsg':
                    # Extract scan data & metadata
                    field_dict = get_tsg_metadata(path.joinpath(unz_path))
                    # Write a row: provider, filename, TSG modified date & TSG metadata fields
                    csvwriter.writerow([prov, ds.name, tsg_mod_date] + list(field_dict.values()))
                    break



if __name__ == "__main__":
    # Get the CSV filename from the command line
    parser = argparse.ArgumentParser(
                    prog='tsg_harvest',
                    description='Harvests TSG metadata from NCI and creates a CSV file summary',)
    parser.add_argument('csv_filename', help='TSG metadata summary will be output to this file')

    args = parser.parse_args()
    with open(args.csv_filename, 'w') as csv_file:
        csvwriter = csv.writer(csv_file, delimiter='|', quotechar='|', doublequote=False,
                                         quoting=csv.QUOTE_NONE)
        # Write CSV header
        csvwriter.writerow(['provider', 'file name', TSG_PUBLISH_DATE] + TSG_FIELDS)
        # Loop over providers
        for prov in PROVIDERS:
            process_prov(prov, csvwriter)
