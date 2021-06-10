# NVCL Reporting

Shane Mule has written some Python scripts to report status of NVCL datasets and services

I will modify them and augment them over time.


```
git clone https://github.com/vjf/nvcl_reporting.git
cd nvcl_reporting
mkdir venv
python3 -m venv ./venv
. ./venv/bin/activate
pip3 install --upgrade pip
pip3 install -r requirements.txt

# Add email address
# First line is the "To:" address
# Second line is the "From:" address
vi .email_addr

# Send off annual email report
./run_reports.sh A

# Send off quarterly email report
./run_reports.sh Q

# Send off weekly email report
./run_reports.sh W
```
