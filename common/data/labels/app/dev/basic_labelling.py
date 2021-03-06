# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_player as player
import argparse

import cv2
import sys
import os
from pathlib import Path

import pandas as pd
import sqlite3
# Add the git root directory to python path
sys.path.insert(0,os.getcwd())
app_file_parent_path = Path(__file__).absolute().parent

import matplotlib.pyplot as plt

# Custom Scripts:
import common.data.labels.app.label_utils as lu
import common.data.labels.generate_index as gi
import common.data.labels.frame_label_utils as flu
from common.data.labels.app.config_utils import JSONPropertiesFile

### TEMPORARY FIXES  ###
# Labels (move to a configuration file created on startup)
labels = [
    'No Danger',
    'Danger'
]

# dummy video
dummy_url = "static/1.mp4"


# Default fields for the application (move to a configuration file created on startup)
default_properties = {
    'current_video_pos'   : 0,
    'current_video_url'   : "",
    'next_footage_step'   : 1,
    'video_urls_file_loc' : '',
    "frame_urls_file_loc" : "",
    "frame_db_loc"        : "",
    "last_frame"          : 0,
    "last_video_url"      : os.path.join(app_file_parent_path, dummy_url),
    "last_label"          : labels[0],
    "current_framerate"   : 0
}

# Set the column name for VIDEO_URLS_FILE
COLNAME = "gap_video_locs"

### \TEMPORARY FIXES ###

# initiate the argument parser
parser = argparse.ArgumentParser()
parser.add_argument("STATIC",
        help="Full location of folder containing the videos we want to label")
parser.add_argument("--PLAYBACK_RATE",
        help="Playback speed of the video player.",
        default=10)
# read arguments from the command line
args = parser.parse_args()

# Add Static Files
# Dash searches for a 'static' file in same folder as this .py file
STATIC_SHORTCUT_LOC = os.path.join(app_file_parent_path, "static")
try:
    os.symlink(args.STATIC, STATIC_SHORTCUT_LOC)
except FileExistsError:
    # refresh the shortcut in case destination has changed
    os.remove(STATIC_SHORTCUT_LOC)
    os.symlink(args.STATIC, STATIC_SHORTCUT_LOC)


# Add Configuration
app_file_parent_path = Path(__file__).absolute().parent
CONFIG_LOC = os.path.join(app_file_parent_path, "config")
CONFIG_FILE_LOC = os.path.join(CONFIG_LOC, "config.json")

# Ensure the configuration file exists
try:
    # generate configuration file
    os.mkdir(CONFIG_LOC)
    print("Directory {} created.".format(CONFIG_LOC))
    print("{} did not exist. \nIt will now be created.".format(CONFIG_LOC, CONFIG_FILE_LOC))
except:
    print("Directory {} already exists.".format(CONFIG_LOC))
    try:
        file = open(CONFIG_FILE_LOC, 'r')
        print("{} already exists.".format(CONFIG_FILE_LOC))
    except IOError:
        print("{} did not exist but the {} directory did. \nIt will now be created.".format(CONFIG_LOC, CONFIG_FILE_LOC))

config_file = JSONPropertiesFile(CONFIG_FILE_LOC, default_properties)
config = config_file.get() 
print("Config read successful.")

# Ensure video_urls.csv exists
VIDEO_URLS_FILE_LOC = os.path.join(app_file_parent_path, "video_urls.csv")

# ensure frames_urls.csv exists
FRAME_URLS_FILE_LOC = os.path.join(app_file_parent_path, "frame_urls.csv")

# Ensure the sqlite database for labels exists
FRAMES_DB = os.path.join(app_file_parent_path, "frames.db")
connex = sqlite3.connect(FRAMES_DB, check_same_thread=False)  # Opens file if exists, else creates file
cur = connex.cursor()                # Send messages and receive results
table_name = "data"                  # Specify the table name

# Get CSS stylesheets
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    html.H1(children='Hello Dash'),

    html.Div(children='''
        Dash: A web application framework for Python.
    '''),
    html.Div(
        className='control-element',
        children=[
                    html.Button('Next Video', id='dropdown-footage-next'),
                    html.Div([
                                dcc.Slider(
                                    id='next_footage_step',
                                    min=-100,
                                    max=100,
                                    step=1,
                                    value=1,
                                    marks={
                                            -100: '-100 Videos',
                                            -10: '-10',
                                            -1: '-1',
                                            1: '1',
                                            10: '10',
                                            100: '100 Videos'
                                        },
                                ),
                                html.Div(id='slider-output-container')
                            ]),
        ]
    ),
    
    html.Div(
                    className='video-outer-container',
                    children=html.Div(
                        style={'width': '100%', 'paddingBottom': '56.25%', 'position': 'relative'},
                        children=player.DashPlayer(
                            id='video-display',
                            style={'position': 'absolute', 'width': '100%',
                                   'height': '100%', 'top': '0', 'left': '0', 'bottom': '0', 'right': '0'},
                            url=dummy_url,
                            controls=True,
                            playing=False,
                            volume=1,
                            width='100%',
                            height='100%',
                            playbackRate= int(args.PLAYBACK_RATE)
                        )
                    )
            ),
    html.Div(
                    className='label-container',
                    children=[
                        html.Div([
                                    dcc.RadioItems(
                                        id='label-radio',
                                        options=[{'label':i, 'value':i} for i in labels],
                                        value=labels[0]
                                    ),  
                                    html.Div(id='dd-output-container')
                                ]),
                        html.Div(id='label-select-output-container')
                    ]
            )
])

# Data Loading
@app.server.before_first_request
def load_all():
    '''
    Perform the following initial steps:
    - Find and update csv file containing urls of all videos in args.STATIC
        - create it if it does not exist
    - Find csv file for storing frame labels
        - create it if it does not exist
    '''
    # load application configuration
    # generate the index of videos to be used
    gi.generate_index(STATIC_SHORTCUT_LOC, VIDEO_URLS_FILE_LOC, COLNAME)
    # generate the index of frames with labels from videos in the index.
    flu.generate_frame_labels(VIDEO_URLS_FILE_LOC,
                              COLNAME,
                              FRAME_URLS_FILE_LOC,
                              PAD=7)
    # setup the SQLite database for read/writes of labels from the tool
    for chunk in pd.read_csv(FRAME_URLS_FILE_LOC, chunksize=1024**2):
        chunk.to_sql(name=table_name,
                     con=connex,
                     if_exists="append",
                     index=False)
    connex.commit()
    # save configuration
    config_file = JSONPropertiesFile(CONFIG_FILE_LOC, default_properties)
    config = config_file.get()
    config["video_urls_file_loc"] = VIDEO_URLS_FILE_LOC
    config["frame_urls_file_loc"] = FRAME_URLS_FILE_LOC
    config["frame_db_loc"]        = FRAMES_DB
    config["last_video_url"]      = config["current_video_url"]
    config_file.set(config)

    

# Footage Update Step
@app.callback(
    Output('slider-output-container', 'children'),
    [Input('next_footage_step', 'value')])
def update_footage(value):
    # Initialise config file
    config_file = JSONPropertiesFile(CONFIG_FILE_LOC, default_properties)
    config = config_file.get()
    # write the new value to the config
    config["next_footage_step"] = value
    config_file.set(config)
    # Display the current selected step
    return 'The selection will move {} steps when "NEXT VIDEO" pressed.'.format(value)

    
# Footage Selection
@app.callback(Output("video-display", "url"),
              [Input('dropdown-footage-next', 'n_clicks')],
              [State('video-display', 'currentTime')])
def next_footage(footage, current_time):
    # Find desired footage and update player video
    # find current video position and step to move
    config_file = JSONPropertiesFile(CONFIG_FILE_LOC, default_properties)
    config = config_file.get()
    current_pos = config["current_video_pos"]
    next_footage_step = config["next_footage_step"]
    url_df = pd.read_csv(config["video_urls_file_loc"])
    new_pos = (current_pos + next_footage_step ) % (len(url_df))
    # find the video corresponding to new_pos
    full_url = url_df.at[new_pos, COLNAME]
    # must change so that it only refers to the static folder (limitation of Dash)
    url = full_url.replace(str(app_file_parent_path), '')
    # get framerate
    video = cv2.VideoCapture(full_url)
    FRAMERATE = int(video.get(cv2.CAP_PROP_FPS))
    # update config
    config["current_video_pos"] = new_pos
    config["current_framerate"] = FRAMERATE
    config["current_video_url"] = full_url
    last_frame        = config["last_frame"]
    last_label        = config["last_label"]
    last_video_url    = config["last_video_url"]
    current_video_url = config["current_video_url"]
    framerate         = config["current_framerate"]
    current_label     = last_label # Did not change the label!
    # get the current frame
    if current_time:
        current_frame = int(round(current_time * framerate))
        if last_frame <= current_frame:
            # update the sql fields
            print("Updating Video: {} \nFrames: {} to {}\nLabel: {}".format(current_video_url, last_frame, current_frame, last_label ))
    else:
        current_frame = 0
        # We are at the start of the video so do nothing
    # set last_frame, last_label, last_video_url now
    config["last_frame"]     = current_frame
    config["last_label"]     = current_label
    config["last_video_url"] = current_video_url
    config_file.set(config)
    # return new url
    return url

# Update label for this scene
@app.callback(
    Output('dd-output-container', 'children'),
    [Input('label-radio', 'value')],
    [State('video-display', 'currentTime')])
def update_label(current_label, current_time):
    '''
    This function is called when the label choice changes.
    To simplify operation, we will only update labels when the video is playing forwards
    ie.
        - if last_frame < current_frame and last_video_url = current_video_url:
            update the label for all frames: last_frame<= frame <current_frame
            to last_label
        - else:
            Do NOT write to database!
    '''
    print(current_time)
    config_file = JSONPropertiesFile(CONFIG_FILE_LOC, default_properties)
    config = config_file.get()
    last_frame        = config["last_frame"]
    last_label        = config["last_label"]
    last_video_url    = config["last_video_url"]
    current_video_url = config["current_video_url"]
    framerate         = config["current_framerate"]
    # get the current frame
    if current_time:
        current_frame = int(round(current_time * framerate))
        if (last_frame <= current_frame) and (current_video_url == last_video_url) and (current_label != last_label):
            # update the sql fields
            print("Updating Video: {} \nFrames: {} to {}\nLabel: {}".format(current_video_url, last_frame, current_frame, last_label ))
    else:
        current_frame = 0
        # We are at the start of the video so do nothing
    # set last_frame, last_label, last_video_url now
    config["last_frame"]     = current_frame
    config["last_label"]     = current_label
    config["last_video_url"] = current_video_url
    config_file.set(config)
    return 'This scene will be labelled: "{}"'.format(current_label)

if __name__ == '__main__':
    app.run_server(debug=True)