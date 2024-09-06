#!/bin/bash -x
#
# Main script to run reports
#
# * Annual run: ./run_reports.sh A
# * Quarterly run: ./run_reports.sh Q
# * Weekly run: ./run_reports.sh W
#
# Go to reporting directory
cd $HOME/dev/gitlab/nvcl_reporting

# Check email
if [ ! -e .email_addr ]; then
echo "Missing 2 line '.email_addr' file with email addresses inside!"
echo "First line is 'To:' address, second is 'From:'"
exit 1
fi

# Activate Python env
eval $(pdm venv activate)

# Run TSG harvest
./src/tsg_harvest/tsg_harvest.py

#
# Run reports
#
if [ "$1" = "A" ]; then
# Takes a while, only run once a year (midnight June 30)
echo "Annual run"
./src/make_reports.py -uf
REPORT_NAME=report.pdf

elif [ "$1" = "Q" ]; then
# Takes a while, only run at end of quarter (Mar 31 etc.)
echo "Quarterly run"
./src/make_reports.py -uf
REPORT_NAME=report.pdf

elif [ "$1" = "W" ]; then
# Run every week, accumulates boreholes created in the past week
echo "Weekly run"
./src/make_reports.py -ub
REPORT_NAME=report-brief.pdf

else
# Print usage if no valid params
echo "Usage: run_reports.sh [A|Q|W]"
exit 1
fi

# Send reports via email
# First address is 'To:' address, second is 'From:'
TO_ADDR=`head -1 .email_addr`
FROM_ADDR=`tail -1 .email_addr`
echo " Here is NVCL Report  " | mutt -s 'NVCL Report' -a $REPORT_NAME $TO_ADDR
exit 0
