# NVCL Reporting

Some Python scripts to generate PDF reports and send emails about the status of NVCL datasets and services

### Setup
```
git clone https://gitlab.com/csiro-geoanalytics/auscope/nvcl_reporting.git
cd nvcl_reporting
mkdir venv
python3 -m venv ./venv
. ./venv/bin/activate
pip3 install --upgrade pip
pip3 install -r requirements.txt

# Add two email addresses to a text file
# First line are the "To:" addresses
# Second line is the "From:" address
# Use a comma with no spaces to separate multiple email addresses
vi .email_addr

# Make directories for extracts and data files
mkdir pkl-weekly
mkdir pkl-yearly
mkdir pkl-quarterly
```

### Create PDF reports (without connecting to NVCL services or altering data & extract files)
```
# Create yearly report
./run_NVCL.py -p -d pkl-yearly

# Create brief weekly report
./run_NVCL.py -b -d pkl-weekly
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
./run_NVCL.py -rsp -d pkl-yearly

# Weekly report (run this once a week)
./run_NVCL.py -rsb -d pkl-weekly
```

### Create extracts (used for comparing with previous points in time)
```
# Create yearly extracts (run this once a year)
./run_NVCL.py -e -d pkl-yearly/

# Create quarterly extracts (run this once a quarter)
./run_NVCL.py -e -d pkl-quarterly/
```

### DB Format

There is only one table, it has the following fields:

1.	Report Category  e.g. "log1", "log2" 
2.	Provider e.g. "tas" "nsw" etc.
3.	Nvcl_ID "10026"
4.	Create_datetime (approximation)
5.	Log_id  "41679f23-ca82-45a2-bbaf-81fb175c808"
6.	Algorithm "Grp1 uTSAS", "Grp1 sjCLST" etc.
7.	Log_type "1"
8.	AlgorithmID "109"
9.	Minerals e.g. ["KAOLIN", "WHITE-MICA"]
10.	Mineral total counts e.g.  [1, 279]
11.	Mineral counts at each depth e.g. [[0.5, {"className": "", "classCount": 36, "classText": "WHITE-MICA", "colour": [1.0, 1.0, 0.0, 1.0]}], [1.5, {"className": "", "classCount": 35, "classText": "WHITE-MICA", "colour": [1.0, 1.0, 0.0, 1.0]}], [2.5, {"className": "", "classCount": 45, "classText": "WHITE-MICA", "colour": [1.0, 1.0, 0.0, 1.0]}], [3.5, {"className": "", "classCount": 58, "classText": "WHITE-MICA", "colour": [1.0, 1.0, 0.0, 1.0]}], ...



