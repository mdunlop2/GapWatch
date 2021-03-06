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
from sqlite3 import Error

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
    Attempt to update each frame that is in between frame_start and frame_end.
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
    
    cursor.execute(sql_update_query, frame_ids)
    print("Table: {} updated!".format(table_name))
    connex.commit()

def insert_label( connex,
                  table_name,
                  colnames,
                  colvals):
    '''
    INPUTS
    connex
    table_name
    colnames
    colvals

    DOES:
    Writes the column values (colvals) to the database table (table_name)
    according to their respective column names (colnames)

    Thus changes to the database model should not require a change to this function
    '''
    cursor = connex.cursor()
    sql = ''' INSERT INTO "{}"({})
              VALUES({}) '''.format(table_name,
                                    ','.join(colnames),
                                    ','.join(['?']*len(colnames)))
    try:
        cursor.execute(sql, colvals)
        connex.commit()
        print("Successfully written to table: {} \nLast row: {}".format(table_name,
                                                                        cursor.lastrowid))
    except Error as e:
        print(e)
        

def read_label(connex,
               table_name,
               frame_id,
               frame_col_name,
               label_col_name):
    cursor = connex.cursor()
    sql = """SELECT {} FROM {} WHERE {} = "{}" """.format(label_col_name,
                                                   table_name,
                                                   frame_col_name,
                                                   frame_id)  
    cursor.execute(sql)  # The cursor gives this command to our DB, the results are now available in cur
    print(cursor.fetchall())
