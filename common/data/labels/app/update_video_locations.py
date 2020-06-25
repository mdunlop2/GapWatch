'''
Helper script when moving videos to new storage.

The .db file contains the exact location of the video,
however we don't want to have to label all of the data again.

Instead, simply change the location in the database.

I found it was easiest to do this by exporting database to .csv file
with DB SQLite (https://sqlitebrowser.org/dl/) and then read into
pandas where a regex could be performed.

USAGE:
python \
    common/data/labels/app/update_video_locations.py \
    -O "/home/mdunlop/Documents/GapWatch/GapWatch/common/data/labels/app/old_SQL_table.csv" \
    -N "/home/mdunlop/Documents/GapWatch/GapWatch/common/data/labels/app/new_SQL_table.csv" \
    -OD "/home/matthew/Documents/GapWatch/common/data/labels/app/static/" \
    -ND "/home/matthew/Documents/GapWatch/GapWatch/common/data/labels/app/static/" \
    -CN 0

'''

import pandas as pd
import argparse
import os, sys
from pathlib import Path

# Add the git root directory to python path
sys.path.insert(0,os.getcwd())
app_file_parent_path = Path(__file__).absolute().parent

if __name__ == '__main__':
    # initiate the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-O",
            help="Location of the old table, stored in a csv file")
    parser.add_argument("-N",
            help="Location of the new table, to be stored in a csv file")
    parser.add_argument("-OD",
            help="Old directory of the video files")
    parser.add_argument("-ND",
            help="New directory of the video files")
    parser.add_argument("-CN",
            help="Which table column is it? Indexed from left, start at 0")
    # read arguments from the command line
    args = parser.parse_args()

    old_df = pd.read_csv(args.O)
    # use a simple string replace
    old_df.iloc[:,int(args.CN)] = old_df.iloc[:,int(args.CN)].str.replace(args.OD, args.ND, regex=False)

    print("New video urls will look like this! \n{}".format(old_df.iloc[0,int(args.CN)]))

    # write to csv now
    old_df.to_csv(args.N, index=False)
    

