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
vid_url = "static/output.mp4"

### \TEMPORARY FIXES ###

app.layout = html.Div(children=[
    html.H1(children='Hello Dash'),

    html.Div(children='''
        Dash: A web application framework for Python.
    '''),
    html.Img(
                id="body-image",
                className="three columns"
    ),
    html.Button('Submit', id='button'),
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
                            height='100%'
                        )
                    )
            ),
])

@app.callback(Output("body-image", "src"),
             [Input('button', 'value')])
def update_image(hover_data):
    # iterate to next image
    video_loc = "/mnt/other/projects/GapWatch/week_1/6 sept Friday/MOVI0000.avi"
    cap = cv2.VideoCapture(video_loc)
    frame_np = lu.get_image(cap, 1)
    fig, ax = plt.subplots(1,1)
    ax.imshow(frame_np)
    out_url = lu.fig_to_uri(fig)

    return out_url



if __name__ == '__main__':
    app.run_server(debug=True)