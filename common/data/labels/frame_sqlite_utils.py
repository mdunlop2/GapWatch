'''
Utilities for dealing with labelling frames of videos at high speed
while the video is playing.
'''

# include standard modules
import argparse
import os.path
import pandas as pd
from pathlib import Path
import time


# video tools
import cv2

# data tools
import sqlite3

def initialise_db(df, connex, name="data"):
    '''
    Takes a pandas dataframe and creates a SQLite3 database
    INPUTS:
    df     - Pandas Dataframe
    connex - sqlite3.connect object
    name   - name of the Table object generated
    '''
    df.to_sql(name=name,
              con=connex,
              if_exists="append",
              index=False)
    