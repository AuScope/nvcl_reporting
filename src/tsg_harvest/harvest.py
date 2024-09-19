#!/usr/bin/env python3
import os
from zipfile import ZipFile
import datetime
import configparser
import csv
import sys
from pathlib import Path, PurePath, PureWindowsPath


import threddsclient
import requests


# Directory where TSG files will be written to
DOWNLOAD_DIR = "/datasets/mr-geochem/work/nci_ncvl_collection"

# URL of NVCL TSG datasets
THREDDS_CAT = 'https://thredds.nci.org.au/thredds/catalog/rs07/{prov}/catalog.xml'

# NVCLDataServices providers
PROVIDERS = ['NSW', 'Vic', 'Tas', 'Qld', 'CSIRO', 'SA', 'NT', 'WA']

# CSV fields extracted from a TSG file
TSG_FIELDS = ['domain id', 'imagelog id', 'proflog id', 'traylog id', 'seclog id', 'drillhole name', 'dataset name', 'lat lon', 'collar']

def parse_date(date_str: str) -> datetime:
    """
    Parse a date string

    :param date_str: date string
    :returns: datetime object, if string can't be parsed returns a datetime of the year 9999
    """
    try:
        dt = datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
    except ValueError as ve:
        print(f"Cannot parse date '{date_str}': {ve}")
        return datetime.datetime.max
    return dt


def get_tsg_scan_dt(filepath: Path) -> datetime:
    """ Reads scan date from TSG file

    :param filename: filename of TSG file
    :returns: datetime or datetime.max upon error
    """
    print(f"Finding scan date in {filepath}")
    config = configparser.ConfigParser(allow_no_value=True, strict=False)
    try:
        # utf-8
        config.read(str(filepath))
    except UnicodeDecodeError:
        # cp1252
        print(f"Cannot read {filepath} using 'utf-8' coding, now trying 'cp1252'")
        config.read(str(filepath), encoding='cp1252')
    # Look for scan date
    try:
        date_str = config['dbextra']['scan date']
    except (configparser.Error, KeyError):
        print(f"{filepath} does not have scan date")
        # Look for create date
        date_str = config['description']['Created']
    print(f"Found {date_str}")
    field_dict =  {'scan date': parse_date(date_str) }
    for field in ['domain id', 'imagelog id', 'proflog id', 'traylog id', 'seclog id', 'drillhole name', 'dataset name', 'lat lon', 'collar']:
        try:
            field_dict[field] = config['dbextra'][field]
        except (configparser.Error, KeyError):
            print(f"{filepath} does not have {field}")
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
        zip_file = os.path.join(prov_dir, ds.name)
        # Has TSG ZIP file been downloaded?
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
            tsg_path = [ uzp for uzp in unzip_paths if str(uzp)[-8:] == '_tsg.tsg' ]
            if len(tsg_path) > 0 and not path.joinpath(tsg_path[0]).exists():
                print(f"Unzipping {zip_file}")
                # This forces 'Zipfile' to create POSIX paths using the Windows paths in the ZIP file
                os.path.altsep = '\\'
                z.extractall(path=prov_dir)

            # Continue to next loop if there is no TSG file
            if not path.joinpath(tsg_path[0]).exists():
                print("ERROR - Cannot find TSG file in {zip_file}")
                continue

            # Extract scan date from TSG file
            for unz_path in unzip_paths:
                if str(unz_path)[-8:] == '_tsg.tsg':
                    field_dict = get_tsg_scan_dt(path.joinpath(unz_path))
                    csvwriter.writerow([prov, ds.name] + list(field_dict.values()))
                    break



if __name__ == "__main__":
    with open("metadata.csv", 'w') as csv_file:
        csvwriter = csv.writer(csv_file, delimiter='|', quotechar='|', doublequote=False,
                                         quoting=csv.QUOTE_NONE)
        # Write CSV header
        csvwriter.writerow(['provider', 'file name', 'scan date'] + TSG_FIELDS)
        # Loop over providers
        for prov in PROVIDERS:
            process_prov(prov, csvwriter)
