#!/bin/bash

if [ "$1" = "A" ]; then
echo "Annual run"
./run_NVCL.py -rsp -d pkl.yearly

elif [ "$1" = "Q" ]; then
echo "Quarterly run"
\rm -rf pkl.weekly
./run_NVCL.py -rsp -d pkl.weekly

elif [ "$1" = "W" ]; then
echo "Weekly run"
./run_NVCL.py -rsp -d pkl.weekly

else
echo "Usage: run_reports.sh [A|Q|W]"
exit 1
fi

# Email
if [ ! -e .email_addr ]; then
echo "Missing 2 line '.email_addr' file with email addresses inside"
echo "First line is 'To:' address, second is 'From:'"
exit 1
fi
# First address is 'To:' address, second is 'From:'
TO_ADDR=`head -1 .email_addr`
FROM_ADDR=`tail -1 .email_addr`
echo "   " | mail -r $FROM_ADDR -s 'NVCL Report' -A report.pdf $TO_ADDR
