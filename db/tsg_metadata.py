#!/usr/bin/env python3

import csv

class TSGMeta:
    def __init__(self, filename: str):
        self.dt_lkup = {}
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

if __name__ == "__main__":
    t = TSGMeta("metadata.csv")
    print(t.dt_lkup)

