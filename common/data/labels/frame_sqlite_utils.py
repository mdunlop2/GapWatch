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

def update_label(connex,
                 table_name,
                 label_col_name,
                 frame_col_name,
                 video_url,
                 frame,
                 new_label,
                 pad = 7):
    '''
    Attempt to update each record one at a  time.
    '''
    frame_id = "{}_{}".format(video_url, str(frame).rjust(pad, '0'))
    cursor = connex.cursor()
    sql_update_query = """Update {} set {} = "{}" where {} = "{}" """.format(table_name,
                                                                         label_col_name,
                                                                         new_label,
                                                                         frame_col_name,
                                                                         frame_id)
    
    print(sql_update_query)
    cursor.execute(sql_update_query)
    
    connex.commit()

def update_label_array( connex,
                        table_name,
                        label_col_name,
                        frame_col_name,
                        video_url,
                        frame_start,
                        frame_end,
                        new_label,
                        pad = 7):
    '''
    Attempt to update each frame that is in our between frame_start and frame_end.
        - Give it label new_label
    
    To do this, we need some funky SQLite code as cannot simply export a list
    to the SQLite executor. Instead, we write a "array" of (?,...,?) and supply the
    arguments.

    This is crude but more efficient than reading the entire dataframe in pandas and
    writing it again.
    '''
    frame_ids = ["{}_{}".format(video_url, str(frame).rjust(pad, '0')) for frame in range(frame_start, frame_end)]
    cursor = connex.cursor()
    sql_update_query = """Update {} set {} = "{}" where {} in ({}) """.format(table_name,
                                                                         label_col_name,
                                                                         new_label,
                                                                         frame_col_name,
                                                                         ','.join(['?']*len(frame_ids)))
    
    print(sql_update_query)
    cursor.execute(sql_update_query, frame_ids)
    print("Table: {} updated!".format(table_name))
    connex.commit()