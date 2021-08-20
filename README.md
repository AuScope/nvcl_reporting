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
# First line is the "To:" address
# Second line is the "From:" address
vi .email_addr

# Make directories for extracts
mkdir pkl-weekly
mkdir pkl-yearly
mkdir pkl-quarterly
```

### Email PDF reports
```
# Send off annual email report
./run_reports.sh A

# Send off quarterly email report
./run_reports.sh Q

# Send off weekly email report
./run_reports.sh W
```

### Create extracts
```
# Create yearly extracts
./run_NVCL.py -e -d pkl-yearly/

# Create quarterly extracts
./run_NVCL.py -e -d pkl-quarterly/
```

### Create PDF reports
```
# Create yearly report
./run_NVCL.py -p -d pkl-yearly

# Create brief weekly report
./run_NVCL.py -pb -d pkl-weekly
```

