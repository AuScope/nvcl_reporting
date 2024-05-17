from pathlib import Path
import datetime

"""
Various shared constants
"""
# NVCL provider list. Format is (WFS service URL, NVCL service URL, bounding box coords)
PROV_LIST = ['NSW', 'NT', 'TAS', 'VIC', 'QLD', 'SA', 'WA']

# Configuration file
CONFIG_FILE = "config.yaml"

# Test run
TEST_RUN = True

# Maximum number of boreholes to retrieve from each provider
MAX_BOREHOLES = 999999

# Abort information file - contains the NVCL log id at which run was aborted
ABORT_FILE = Path('./run_NVCL_abort.txt')

# Report data categories
DATA_CATS = ['log1', 'log2', 'log6', 'empty', 'nodata']

# Borehole parameters
HEIGHT_RESOLUTION = 1.0
ANALYSIS_CLASS = ''

# Report time window is based on this date
REPORT_DATE = datetime.datetime(2020, 10, 17)
