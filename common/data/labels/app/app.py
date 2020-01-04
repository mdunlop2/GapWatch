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
# Add the git root directory to python path
sys.path.insert(0,os.getcwd())

import matplotlib.pyplot as plt

# Custom Scripts:
import common.data.labels.app.label_utils as lu

# initiate the parser
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

# Get CSS stylesheets
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

### TEMPORARY FIXES  ###
# attempt to use an array of urls
vid_url = "static/output.mp4"

### \TEMPORARY FIXES ###

app.layout = html.Div(children=[
    html.H1(children='Hello Dash'),

    html.Div(children='''
        Dash: A web application framework for Python.
    '''),
    html.Div(
        className='control-element',
        children=[
                    html.Button('Next Video', id='dropdown-footage-next'),
                    html.Button('Previous Video', id='dropdown-footage-prev'),
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
                            url=vid_url,
                            controls=True,
                            playing=False,
                            volume=1,
                            width='100%',
                            height='100%',
                            playbackRate= args.PLAYBACK_RATE
                        )
                    )
            ),
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
    
# Footage Selection
@app.callback(Output("video-display", "url"),
              [Input('dropdown-footage-next', 'value')])
def select_footage(footage, display_mode):
    # Find desired footage and update player video
    url = url_dict[display_mode][footage]
    return url


if __name__ == '__main__':
    app.run_server(debug=True)