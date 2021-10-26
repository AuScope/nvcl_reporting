#!/bin/bash
#
# Main script to run reports
#
# * Annual run: ./run_reports.sh A
# * Quarterly run: ./run_reports.sh Q
# * Weekly run: ./run_reports.sh W
#
# Go to reporting directory
cd ~/gitlab/nvcl_reporting

# Check email
if [ ! -e .email_addr ]; then
echo "Missing 2 line '.email_addr' file with email addresses inside"
echo "First line is 'To:' address, second is 'From:'"
exit 1
fi

# Activate Python env
. ./venv/bin/activate

if [ "$1" = "A" ]; then
# Takes a while, only run once a year (midnight June 30)
echo "Annual run"
# TODO: Backup last year
./run_NVCL.py -rsp -d pkl-yearly
./run_NVCL.py -e -d pkl-yearly
REPORT_NAME=report.pdf

elif [ "$1" = "Q" ]; then
# Takes a while, only run at end of quarter (Mar 31 etc.)
echo "Quarterly run"
# TODO: Backup last quarter
./run_NVCL.py -rsp -d pkl-quarterly
./run_NVCL.py -e -d pkl-quarterly
REPORT_NAME=report.pdf

elif [ "$1" = "W" ]; then
# Run every week, accumulates boreholes created in the past week
echo "Weekly run"
./run_NVCL.py -rsb -d pkl-weekly
REPORT_NAME=report-brief.pdf

else
echo "Usage: run_reports.sh [A|Q|W]"
exit 1
fi

# First address is 'To:' address, second is 'From:'
TO_ADDR=`head -1 .email_addr`
FROM_ADDR=`tail -1 .email_addr`
echo "   " | mail -r $FROM_ADDR -s 'NVCL Report' -A $REPORT_NAME $TO_ADDR
exit 0
