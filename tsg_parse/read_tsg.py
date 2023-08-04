#!/usr/bin/env python3

import configparser
from datetime import datetime

def get_tsg_scan_dt(filename: str) -> datetime:
    """ Reads scan date from TSG file 

    :param filename: filename of TSG file
    :returns: datetime or datetime.max upon error
    """
    config = configparser.ConfigParser(allow_no_value=True, strict=False)

    config.read(filename)
    try:
        scan_date_str = config['dbextra']['scan date']
        scan_dt = datetime.strptime(scan_date_str, '%Y-%m-%d %H:%M:%S') 
    except (configparser.Error, KeyError) as ee:
        print(f"Cannot find scan date in {filename}: {ee}")
        return datetime.max
    except ValueError as ve:
        print(f"Cannot parse date '{scan_date_str}': {ve}")
        return datetime.max
    return scan_dt

if __name__ == '__main__':
    assert get_tsg_scan_dt("8440735_11CPD005_tsg.tsg") == datetime(2013, 2, 1, 12, 13, 29)
