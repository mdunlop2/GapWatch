'''
Want to verify that labels are being correctly saved
in the SQL database and trial out ways to read frame labels
'''
# include standard modules
import argparse
import os
import sys
import pandas as pd
from pathlib import Path
import time

# data tools
import sqlite3

# Add the git root directory to python path
sys.path.insert(0,os.getcwd())

import common.data.labels.frame_sqlite_utils as squ

if __name__ == "__main__":
    table_name = "data"
    frame_id = "/home/matthew/Documents/GapWatch/common/data/labels/app/static/10.mp4"
    frame_col_name = "video_url"
    label_col_name = "label"
    FRAMES_DB = "/home/matthew/Documents/GapWatch/common/data/labels/app/frames.db"
    connex = sqlite3.connect(FRAMES_DB, check_same_thread=False)
    squ.read_label(connex,
                   table_name,
                   frame_id,
                   frame_col_name,
                   label_col_name)