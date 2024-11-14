from pathlib import Path
import datetime

# This file's path
file_path = Path( __file__ ).absolute()

# Repo root path
ROOT_PATH = file_path.parents[1]

"""
Various shared constants
"""
# NVCL provider list. Format is (WFS service URL, NVCL service URL, bounding box coords)
PROV_LIST = ['NSW', 'NT', 'TAS', 'QLD', 'SA', 'WA', 'VIC', 'CSIRO']

# Configuration file
CONFIG_FILE = str(ROOT_PATH / "config.yaml")

# Test run
TEST_RUN = False

# Maximum number of boreholes to retrieve from each provider
MAX_BOREHOLES = 999999

# Abort information file - contains the NVCL log id at which run was aborted
ABORT_FILE = Path('./run_NVCL_abort.txt')

# Report data categories
DATA_CATS = ['log0', 'log1', 'log2', 'log3', 'log4', 'log5', 'log6', 'empty', 'nodata']
DATA_CATS_NUMS = ['0', '1', '2', '3', '4', '5', '6']
# Borehole parameters
HEIGHT_RESOLUTION = 1.0 # One metre resolution
ANALYSIS_CLASS = ''     # All classes

# Report time window is based on this date
REPORT_DATE = datetime.datetime.now().date()

# Plot image size
IMAGE_SZ = [150, 100]

# Report font
FONT = 'helvetica'
