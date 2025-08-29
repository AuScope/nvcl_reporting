#!/bin/bash
#
# Main script to run reports
#
# * Annual run: ./run_reports.sh A
# * Quarterly run: ./run_reports.sh Q
# * Weekly run: ./run_reports.sh W
#
# Assumes we're in script directory

# Add pdm to PATH
export PATH=$HOME/.local/bin:$PATH

# Go to repository root directory
cd ..

# Check email file
if [ ! -e .email_addr ]; then
echo "Missing '.email_addr' file with email addresses inside!"
echo "First line is 'To:' addresses separated by spaces"
exit 1
fi

# Activate Python env
eval $(pdm venv activate)

#
# Run reports
#
if [ "$1" = "A" ]; then
# Takes a while, only run once a year (midnight June 30)
echo "Annual run"
FREQUENCY="Annual"
REPORT_NAME=annual-nvcl-report.pdf
\rm -f $REPORT_NAME
./src/make_reports.py -utf -o $REPORT_NAME

elif [ "$1" = "Q" ]; then
# Takes a while, only run at end of quarter (Mar 31 etc.)
echo "Quarterly run"
FREQUENCY="Quarterly"
REPORT_NAME=quarterly-nvcl-report.pdf
\rm -f $REPORT_NAME
./src/make_reports.py -utf -o $REPORT_NAME

elif [ "$1" = "W" ]; then
# Run every week, accumulates boreholes created in the past week
echo "Weekly run"
FREQUENCY="Weekly Brief"
REPORT_NAME=brief-nvcl-report.pdf
\rm -f $REPORT_NAME
./src/make_reports.py -utb -o $REPORT_NAME

else
# Print usage if no valid params
echo "Usage: run_reports.sh [A|Q|W]"
exit 1
fi

if [ -e $REPORT_NAME ]; then
echo "Sending report email"
# First line contains 'To:' addresses
TO_ADDR=`head -1 .email_addr`
# Send reports via email using mutt
echo "Please find attached $FREQUENCY NVCL Report" | mutt -s "$FREQUENCY NVCL Report" -a $REPORT_NAME -- $TO_ADDR
exit 0

else
echo "Report file $REPORT_NAME was not generated. Exiting."
exit 1
fi
