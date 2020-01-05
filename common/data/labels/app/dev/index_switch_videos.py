# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_player as player
import argparse

import cv2
import sys
import os
from pathlib import Path

import pandas as pd
# Add the git root directory to python path
sys.path.insert(0,os.getcwd())

import matplotlib.pyplot as plt

# Custom Scripts:
import common.data.labels.app.label_utils as lu
import common.data.labels.generate_index as gi
from common.data.labels.app.config_utils import JSONPropertiesFile

### TEMPORARY FIXES  ###
# attempt to use an array of urls
url_list = ["static/output.mp4",
            "static/1_CPU.mp4",
            "static/2_CPU.mp4"]

# Labels (move to a configuration file created on startup)
labels = [
    'No Danger',
    'Danger'
]


# Default fields for the application (move to a configuration file created on startup)
default_properties = {
    'current_video_pos' : 0,
    'next_footage_step' : 1,
    'current_scene_label' : labels[0],
    'video_urls_file_loc' : ''
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
app_file_parent_path = Path(__file__).absolute().parent
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
                            url="static/1.mp4",
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
    config_file = JSONPropertiesFile(CONFIG_FILE_LOC, default_properties)
    config = config_file.get()
    config["video_urls_file_loc"] = VIDEO_URLS_FILE_LOC
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
              [Input('dropdown-footage-next', 'n_clicks')])
def next_footage(footage):
    # Find desired footage and update player video
    # find current video position and step to move
    config_file = JSONPropertiesFile(CONFIG_FILE_LOC, default_properties)
    config = config_file.get()
    current_pos = config["current_video_pos"]
    next_footage_step = config["next_footage_step"]
    url_df = pd.read_csv(config["video_urls_file_loc"])
    new_pos = (current_pos + next_footage_step ) % (len(url_df))
    # find the video corresponding to new_pos
    # must change so that it only refers to the static folder (limitation of Dash)
    url = url_df.at[new_pos, COLNAME].replace(str(app_file_parent_path), '')
    # this url needs to be changed to its location in STATIC
    # update config
    config["current_video_pos"] = new_pos
    config_file.set(config)
    # return new url
    return url

# Update label for this scene
@app.callback(
    Output('dd-output-container', 'children'),
    [Input('label-radio', 'value')])
def update_output(value):
    return 'This scene will be labelled: "{}"'.format(value)

if __name__ == '__main__':
    app.run_server(debug=True)