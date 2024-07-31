#!/usr/bin/env python3

import csv
import os
import sys

class TSGMeta:
    def __init__(self, filename: str):
        self.dt_lkup = {}
        if os.path.exists(filename):
            with open(filename) as csvfile:
                tsg_reader = csv.reader(csvfile, delimiter='|')
                first = True
                for row in tsg_reader:
                    if first:
                        first = False
                        continue
                    for col in row:
                        # Lookup table - associate borehole name <==> scan date
                        self.dt_lkup[row[8]] = row[2]
        else:
            print(f"ERROR - Cannot find TSG metadata file: {filename}")
            sys.exit(1)

if __name__ == "__main__":
    # For testing 
    t = TSGMeta("metadata.csv")
    print(t.dt_lkup)

