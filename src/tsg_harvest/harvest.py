#!/usr/bin/env python3
import os
import shutil
from zipfile import ZipFile, BadZipFile
import datetime
import configparser
import csv
import sys
import tempfile
from pathlib import Path, PurePath, PureWindowsPath
import argparse
from collections import defaultdict



import threddsclient
from threddsclient.nodes import Dataset
import requests


from helpers import load_and_check_config

# URL of NVCL TSG datasets
THREDDS_CAT = 'https://thredds.nci.org.au/thredds/catalog/rs07/{prov}/catalog.xml'

# NVCLDataServices providers
PROVIDERS = ['NSW', 'Vic', 'Tas', 'Qld', 'CSIRO', 'SA', 'NT', 'WA']

HL_SCAN_DATE = 'hl scan date' # Date of core scan by Hylogger
TSG_PUBLISH_DATE = 'tsg publish date' # Last modified date of TSG file in THREDDS filesystem
NVCL_ID = 'drillhole name'

# CSV fields extracted from a TSG file
TSG_FIELDS = [HL_SCAN_DATE, 'domain id', 'imagelog id', 'proflog id', 'traylog id', 'seclog id', NVCL_ID, 'dataset name', 'lat lon', 'collar']

DATE_FMT = '%Y-%m-%d %H:%M:%S'

def parse_date(date_str: str) -> datetime:
    """
    Parse scan date string take from TSG file

    :param date_str: date string
    :returns: datetime object, if string can't be parsed returns a datetime of the year 9999
    """
    try:
        dt = datetime.datetime.strptime(date_str, DATE_FMT)
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
        print(f"Reading {filepath} as 'cp1252'")
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


def process_prov(prov: str, prov_tsg_dict: dict[str, list]):
    """
    Extract data and process for one provider

    :param prov: provider name - see 'PROVIDERS' list
    :param csvwrite: instance of csvwriter class to output each data point
    :param prov_tsg_dict: TSG dict for prov, key is TSG zip filename, val is remaining fields
    """
    print(f"Calling read_url({THREDDS_CAT.format(prov=prov)})")	
    cat = threddsclient.read_url(THREDDS_CAT.format(prov=prov))

    for ds in cat.flat_datasets():
        # Fetch THREDDS modified date for TSG file
        try:
            tsg_mod_datetime = datetime.datetime.strptime(ds.modified, "%Y-%m-%dT%H:%M:%SZ")
        except ValueError as ve:
            print(f"ERROR - THREDDS provider {prov} file {ds.name} has modified date {ds.modified} is not a valid date: {ve}")
            continue

        # Is there are more recent version of TSG ZIP file?
        if prov_tsg_dict.get(ds.name, [datetime.datetime.min])[0] < tsg_mod_datetime:
            with tempfile.NamedTemporaryFile(mode='wb') as temp_fd:
                print(f"Processing {ds.name}. Downloading {ds.download_url()}")
                # Download TSG ZIP file
                try:
                    r = requests.get(ds.download_url())  
                    r.raise_for_status()
                    temp_fd.write(r.content)
                except requests.exceptions.RequestException as re:
                    print(f"ERROR - error downloading {ds.download_url()}: {re}")
                    continue

                # Process zip file
                process_tsg_zip(temp_fd.name, ds, tsg_mod_datetime, prov_tsg_dict, prov)
        else:
            print(f"Skipping {ds.name} @ {tsg_mod_datetime}")


def process_tsg_zip(temp_zip_file: str, ds: Dataset, tsg_mod_datetime: datetime.datetime,
                    prov_tsg_dict: dict[str, list], prov: str):
    """
    Process TSG files in a zip file 

    :param temp_zip_file: zip file to unpack
    :param ds: threddsclient Dataset object of the zip file
    :param tsg_mod_datetime: THREDDS TSG zip file modification time
    :param prov_tsg_dict: dictionary used to write TSG metadata 
    :param prov: provider (e.g. 'NSW')
    """
        
    # Unzip TSG files
    try:
        with ZipFile(temp_zip_file) as z:
            # Compile a list of the file paths in the zip file
            unzip_paths = []
            # We don't know if this is a POSIX path or a Windows path
            # So this will convert either kind to a POSIX path
            for p in z.namelist():
                pwp = PureWindowsPath(p)
                pp = PurePath(pwp.as_posix())
                unzip_paths.append(pp) 

            # Extract everything in zip file into a temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                path = Path(temp_dir)
                print(f"Unzipping {temp_zip_file}")
                # This forces 'Zipfile' to create POSIX paths using the Windows paths in the ZIP file
                os.path.altsep = '\\'
                z.extractall(path=temp_dir)

                # Extract scan date and other metadata from TSG file
                for unz_path in unzip_paths:
                    # Look for TSG metadata file
                    if str(unz_path)[-8:] == '_tsg.tsg':
                        # Extract scan data & metadata
                        field_dict = get_tsg_metadata(path.joinpath(unz_path))
                        # Write a row: filename -> TSG modified date & TSG metadata fields
                        prov_tsg_dict[ds.name] = [tsg_mod_datetime] + list(field_dict.values())
                        break
    except BadZipFile as bzf:
        print(f"ERROR: For {prov} & {ds.name}, cannot unzip {temp_zip_file}: {bzf}")


def process_tsgs(output_file: str, tsg_dict: dict[str, dict[str, list]]):
    """
    Extract metadata from TSGs given an output file

    :param output_file: CSV file to put all the output
    :param tsg_dict: TSG dict, key1 is provider, key2 is TSG zip filename, val is remaining fields
    """
    # Loop over providers
    for prov in PROVIDERS:
        print(f"\n\n*** Processing {prov} ***\n")
        # NB: No need to check for missing key in dict because it is defaultdict
        process_prov(prov, tsg_dict[prov])


def parse_csv(csv_file: str) -> dict[str, dict[str, list]]:
    """
    Parses the current CSV file into a dict

    :param csv_file: file path of CSV file to read
    :returns: dict, key1 is provider, key2 is TSG zip filename, val is remaining fields
    """
    with open(csv_file, 'r') as csv_fd:
        csvreader = csv.reader(csv_fd, delimiter='|', quotechar='|', doublequote=False,
                                         quoting=csv.QUOTE_NONE)
        tsg_dict = defaultdict(lambda: defaultdict(list))
        first_row = False
        for prov, zip_file, *field_list in csvreader:
            # Skip first row
            if not first_row:
                first_row = True
                continue
            try:
                dt = datetime.datetime.strptime(field_list[0], DATE_FMT)
            except ValueError as ve:
                print(f"ERROR - Cannot parse scan date '{field_list[0]}' from: {prov}, {zip_file}: {ve}")
                continue
            tsg_dict[prov][zip_file] = [dt] + field_list[1:]
    return tsg_dict


def write_csv(csv_file: str, tsg_dict: dict[str, dict[str, list]]):
    """
    Write out dict values into CSV file

    :param csv_file: filename of CSV file
    :param tsg_dict: dict to be written out
    """

    # Open file in overwrite mode
    with open(csv_file, 'w') as csv_fd:
        csvwriter = csv.writer(csv_fd, delimiter='|', quotechar='|', doublequote=False,
                                         quoting=csv.QUOTE_NONE)
        # Write CSV header
        csvwriter.writerow(['provider', 'file name', TSG_PUBLISH_DATE] + TSG_FIELDS)

        # Write dict as rows
        for prov, t1_dict in tsg_dict.items():
            for zip_file, field_list in t1_dict.items():
                # Write a row: provider, filename, TSG modified date & TSG metadata fields
                try:
                    mod_datetime_str = field_list[0].strftime(DATE_FMT)
                except (ValueError, TypeError) as vte:
                    print(f"ERROR - Cannot format date {field_list[0]} for {prov}, {zip_file}")
                    continue
                csvwriter.writerow([prov, zip_file] + list(field_list))
                

def rotate_backups(base_filename: str, num_versions: int):
    """
    Make up to 'num_versions' copies of a file

    :param base_filename: name of file to be copied
    :param num_versions: number of versions to keep
    """

    # Remove the oldest backup if it exists
    oldest = f"{base_filename}.{num_versions}"
    if os.path.exists(oldest):
        os.remove(oldest)

    # Shift backups: backup.4 → backup.5, backup.3 → backup.4, ...
    for i in range(num_versions - 1, 0, -1):
        src = f"{base_filename}.{i}"
        dst = f"{base_filename}.{i + 1}"
        if os.path.exists(src):
            shutil.move(src, dst)

    # Create new backup from base file
    if os.path.exists(base_filename):
        shutil.copy2(base_filename, f"{base_filename}.1")


def process(config):

    # Make a backup of CSV file
    print("Backup current CSV file")
    csv_file = config['tsg_meta_file']
    rotate_backups(csv_file, 10)

    # Read CSV file
    print("Read current CSV file")
    tsg_dict = parse_csv(csv_file)

    # Amend TSG dict
    process_tsgs(csv_file, tsg_dict)

    # Write out TSG dict to CSV file
    print("Write out new CSV file")
    write_csv(csv_file, tsg_dict)


if __name__ == "__main__":
    # Get the config filename from the command line
    parser = argparse.ArgumentParser(
                    prog='tsg_harvest',
                    description='Harvests TSG metadata from NCI and creates a CSV file summary',)
    parser.add_argument('config_file', help='config file, used to specify output CSV file')

    args = parser.parse_args()
    config = load_and_check_config(args.config_file)
    process(config)
