#!/usr/bin/env bash
[ ! -e .email_addr ] && echo "Missing '.email_addr' file with email address inside" && exit 1
EM=`cat .email_addr`
#python3 run_NVCL.py | mail -r $EM -s "NVCL Report" $EM
