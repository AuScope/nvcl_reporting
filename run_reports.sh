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
fi
