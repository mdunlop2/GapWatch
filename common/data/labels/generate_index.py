'''
Generate a CSV file with the locations of all the
videos available in a folder

ARGUMENTS
- FOLDER
    - Folder containing the .mp4 files
    - Try: /mnt/other/projects/GapWatch/week_1
- OUT_LOC
    - Location to output the index csv

CSV DESIRED FIELDS:

- LOC (string)
    - Location of the .mp4 file (Within FOLDER)
'''
# include standard modules
import argparse
import os.path
import pandas as pd
from pathlib import Path
import time

# initiate the parser
parser = argparse.ArgumentParser()

# add arguments
parser.add_argument("FOLDER", help="folder which contains the .mp4 or .avi files. \nMust give full address")
parser.add_argument("OUT_LOC", help="location of the .csv file to store index in. \nMust give full address")
parser.add_argument("--COL_NAME",
                    help="Name of column used in csv to store the file locations",
                    default="gap_video_locs")



# read arguments from the command line
args = parser.parse_args()

colname = args.COL_NAME

# Check if OUT_LOC already exists
# if it does, then load it in and we'll join the new locations to it
# this allows for updating if new videos are added etc

def find_files(root, extensions):
    for ext in extensions:
        yield from Path(root).glob(f'**/*.{ext}')

start = time.time()
gap_video_locs = [str(gap_video) for gap_video in find_files(args.FOLDER, ['mp4', 'avi'])]

print("Time Taken: {}".format(time.time()-start))

gap_video_loc_df = pd.DataFrame(gap_video_locs,
                                columns = [colname])


if os.path.isfile(args.OUT_LOC):
    print ("Output file already exists.")
    current_gap_video_loc_df = pd.read_csv(args.OUT_LOC)
    # concatenate to join both old and new
    concat_gap_video_loc = pd.concat([current_gap_video_loc_df, gap_video_loc_df])\
                                .drop_duplicates()\
                                .reset_index(drop=True)
    concat_gap_video_loc.to_csv(args.OUT_LOC,
                                index = False)
    print("Successfully updated csv file at {}".format(args.OUT_LOC))


else:
    print ("Output file does not exist.")
    gap_video_loc_df.to_csv(args.OUT_LOC,
                        index = False)
    print("Successfully created csv file at {}".format(args.OUT_LOC))