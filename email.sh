#!/usr/bin/env bash
[ ! -e .email_addr ] && echo "Missing '.email_addr' file with email address inside" && exit 1
# First address is 'To:' address, second is 'From:'
TO_ADDR=`head -1 .email_addr`
FROM_ADDR=`tail -1 .email_addr`
echo "   " | mail -r $FROM_ADDR -s 'NVCL Report' -A report.pdf $TO_ADDR
