[![pdm-managed](https://img.shields.io/badge/pdm-managed-blueviolet)](https://pdm.fming.dev)
[![pipeline status](https://gitlab.com/csiro-geoanalytics/auscope/nvcl_reporting/badges/main/pipeline.svg)](https://gitlab.com/csiro-geoanalytics/auscope/nvcl_reporting/-/commits/main) 
[![coverage report](https://gitlab.com/csiro-geoanalytics/auscope/nvcl_reporting/badges/main/coverage.svg)](https://gitlab.com/csiro-geoanalytics/auscope/nvcl_reporting/-/commits/main)


# NVCL Reporting

Creates a summary database of the NVCL core library from around Australia

There are also some Python scripts to generate PDF reports and send emails about the status of NVCL datasets and services

### Setup

1. If you don't already have it, install [pdm](https://pdm.fming.dev/latest/)

2. Clone repository
```
git clone https://gitlab.com/csiro-geoanalytics/auscope/nvcl_reporting.git
cd nvcl_reporting
pdm install
```

3. Set up email. Add a line of email addresses to a text file called '.email_addr' in the root directory
  - First line has the "To:" addresses
  - Use a space to separate multiple email addresses
```
vi .email_addr
```

4. Install [mutt](http://www.mutt.org) email client
```
sudo apt install mutt
```

5. Configure sending address
```
vi ~/.muttrc
```

Add the following:
```
###############
# Identity
#
set realname = "Fred Smith"
set from = "fred.smith@blah.org.au"
```

### Email PDF reports (will update data & extract files)
```
eval $(pdm venv activate)

cd scripts

# Send off annual email report (run this once a year)
./run_reports.sh A

# Send off quarterly email report (run this once a quarter)
./run_reports.sh Q

# Send off weekly email report (run this once a week)
./run_reports.sh W
```

### Sample crontab

Run 1am Sunday morning each week
```
00 01 * * SUN /usr/bin/bash -c "cd $HOME/gitlab/nvcl_reporting/scripts && ./run_reports.sh W > output.weekly 2>&1"
```

### Generate TSG file metadata summary

```
pdm run src/tsg_harvest/harvest.py config.yaml
```



### Connect to NVCL services, update data files, then create PDF reports 
```
eval $(pdm venv activate)

cd src

# Yearly or Quarterly report (run this once a year)
./make_reports.py -uf

# Brief weekly report (run this once a week)
./make_reports.py -ub
```

### DB Format

There is only one table "meas", it has the following fields:

1.	*report_category*  Report Category e.g. "log1", "log2" 
2.	*provider* State or Territory e.g. "tas" "nsw" etc.
3.	*borehole_id* "10026"
4.  *drill_hole_name* Name of drill hole
5.  *hl_scan_date* HyLogger scan date taken from TSG file
6.  *easting* east coordinate in metres
7.  *northing* north coordate in metres
8.  *crs* coordinate reference system e.g. 'EPSG:7842'
9.  *start_depth* borehole starts at this depth (metres)
10. *end_depth* borehole stops at this depth (metres)
11. *has_vnir* True iff borehole has VNIR (Visible and Near Infra Red) data
12. *has_swir* True iff borehole has SWIR (Short Wave Infra Red) data
13. *has_tir* True iff borehole has TIR (Thermal Infra Red) data
14. *has_mir* True iff borehole has MIR (Mid Infra Red) data
15. *nvcl_id* NVCL ID e.g. "10026"
16.	*modified_datetime* (if provided) e.g. '2023/10/30'
17.	*log_id* e.g. "41679f23-ca82-45a2-bbaf-81fb175c808"
18.	*algorithm* e.g. "Grp1 uTSAS", "Grp1 sjCLST" etc.
19.	*log_type* e.g. "1"
20.	*algorithm_id* e.g. "109"
21.	*minerals* e.g. ["KAOLIN", "WHITE-MICA"]
22.	*mincnts* Mineral total counts e.g. [1, 279]
23.	*data* Mineral counts at each depth e.g. [[0.5, {"className": "", "classCount": 36, "classText": "WHITE-MICA", "colour": [1.0, 1.0, 0.0, 1.0]}], [1.5, {"className": "", "classCount": 35, "classText": "WHITE-MICA", "colour": [1.0, 1.0, 0.0, 1.0]}], [2.5, {"className": "", "classCount": 45, "classText": "WHITE-MICA", "colour": [1.0, 1.0, 0.0, 1.0]}], [3.5, {"className": "", "classCount": 58, "classText": "WHITE-MICA", "colour": [1.0, 1.0, 0.0, 1.0]}], ...

### Testing

* Install [tox](https://tox.wiki)

* Run 'tox' from repository root directory

