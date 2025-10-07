[![pdm-managed](https://img.shields.io/badge/pdm-managed-blueviolet)](https://pdm.fming.dev)
[![lint and test](https://github.com/AuScope/nvcl_reporting/actions/workflows/nvcl_reporting.yml/badge.svg)](https://github.com/AuScope/nvcl_reporting/actions/workflows/nvcl_reporting.yml)
[![Coverage Status](https://raw.githubusercontent.com/AuScope/nvcl_reporting/refs/heads/main/test/badge/coverage-badge.svg)]()

# NVCL Reporting

Creates a summary database of the NVCL core library from around Australia

There are also some Python scripts to generate PDF reports and send emails about the status of NVCL datasets and services

**Needs 64 GB RAM**

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


### Generate TSG file metadata summary, connect to NVCL services, update data files, then create PDF reports 
```
eval $(pdm venv activate)

cd src

# Yearly or Quarterly report (run this once a year)
./make_reports.py -utf

# Brief weekly report (run this once a week)
./make_reports.py -utb
```
**NOTES:**
1. Omit 't' command line flag to disable TSG file summary
2. 'u' command line flag updates the current database
3. 'f' command line flag generates a full report
4. 'b' command line flag generates a brief report


### DB Format

There are two tables:
1. "meas" this has TSG metadata, borehole metadata and mineralogy 
2. "stats" this contains statistics e.g. sum of borehole depths  

#### "meas" table

It has the following fields:

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

#### "stats" table

It has the following fields:

1. *stat_name* name of statistic
2. *provider* State or Territory e.g. "tas" "nsw" etc.
3. *start_date* statistic measurement start date
4. *end_date* statistic measurement end date
5. *stat_val1* statistic value 1 (float)
6. *stat_val2* statistic value 2 (float)


### Docker

* Docker compose file is [here](./docker/docker-compose.yml)
* Instructions are [here](./docker/README.md)

### Grafana

* [Grafana](https://github.com/grafana/grafana) is used to display the data as a series of tables and graphs.
* The configuration for Grafana dashboards and data sources are exported to file using [grizzly](https://github.com/grafana/grizzly) and [grafanactl](https://grafana.github.io/grafanactl/)
* They are kept [here](./grafana/grizzly_bkup) and [here](./grafana/grafanactl_bkup)

### Testing

* Install [tox](https://tox.wiki)
* Run 'tox' from repository root directory

