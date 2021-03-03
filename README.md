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
vi .email_addr

# Send off email report
./email.sh
```
