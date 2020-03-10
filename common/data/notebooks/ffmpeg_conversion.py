'''
Executable python file which converts .avi files to .mp4 files
'''

# include standard modules
import argparse
import os.path
import pandas as pd
from pathlib import Path
import time

import pandas as pd

import subprocess

def find_files(root, extensions):
    for ext in extensions:
        yield from Path(root).glob(f'**/*.{ext}')

if __name__ == "__main__":
    # initiate the parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("FOLDER", help="Parent Directory of the .avi files")
    parser.add_argument("OUT_LOC", help="Location of folder to store the new .MP4 files in")
    parser.add_argument("--COL_NAME",
                        help="Name of column used in csv to store the file locations",
                        default="gap_video_locs")
    # read arguments from the command line
    args = parser.parse_args()

    avi_file_paths = [str(gap_video) for gap_video in find_files(args.FOLDER, ['avi'])]
    for i in range(len(avi_file_paths)):
        avi_file = avi_file_paths[i]
        mp4_file = "{}/{}.mp4".format(args.OUT_LOC, i)
        subprocess_template = ["sudo", "ffmpeg", "-i", avi_file, mp4_file]
        subprocess.call(subprocess_template)
        print(avi_file)
