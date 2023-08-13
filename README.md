# NVCL Reporting

Creates a summary database of the NVCL core library from around Australia

There are also some Python scripts to generate PDF reports and send emails about the status of NVCL datasets and services

### Setup

If you don't already have it, install [pdm](https://pdm.fming.dev/latest/)

```
git clone https://gitlab.com/csiro-geoanalytics/auscope/nvcl_reporting.git
cd nvcl_reporting
pdm install

# Add two email addresses to a text file
# First line are the "To:" addresses
# Second line is the "From:" address
# Use a comma with no spaces to separate multiple email addresses
vi .email_addr
```

### Email PDF reports (will update data & extract files)
```
# Send off annual email report (run this once a year)
./run_reports.sh A

# Send off quarterly email report (run this once a quarter)
./run_reports.sh Q

# Send off weekly email report (run this once a week)
./run_reports.sh W
```

### Connect to NVCL services, update data files, then create PDF reports 
```
# Yearly report (run this once a year)
./make_reports.py -usp

# Brief weekly report (run this once a week)
./make_reports.py -usb
```

### DB Format

There is only one table "meas", it has the following fields:

1.	*report_category*  Report Category e.g. "log1", "log2" 
2.	*provider* State or Territory e.g. "tas" "nsw" etc.
3.	*nvcl_id* "10026"
4.	*modified_datetime* (approximation) e.g. '2023/10/30'
5.	*log_id* e.g. "41679f23-ca82-45a2-bbaf-81fb175c808"
6.	*algorithm* e.g. "Grp1 uTSAS", "Grp1 sjCLST" etc.
7.	*log_type* e.g. "1"
8.	*algorithm_id* e.g. "109"
9.	*minerals* e.g. ["KAOLIN", "WHITE-MICA"]
10.	*mincnts* Mineral total counts e.g. [1, 279]
11.	*data* Mineral counts at each depth e.g. [[0.5, {"className": "", "classCount": 36, "classText": "WHITE-MICA", "colour": [1.0, 1.0, 0.0, 1.0]}], [1.5, {"className": "", "classCount": 35, "classText": "WHITE-MICA", "colour": [1.0, 1.0, 0.0, 1.0]}], [2.5, {"className": "", "classCount": 45, "classText": "WHITE-MICA", "colour": [1.0, 1.0, 0.0, 1.0]}], [3.5, {"className": "", "classCount": 58, "classText": "WHITE-MICA", "colour": [1.0, 1.0, 0.0, 1.0]}], ...

