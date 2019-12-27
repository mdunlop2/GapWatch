# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import cv2
import sys
import os
# Add the git root directory to python path
sys.path.insert(0,os.getcwd())

import matplotlib.pyplot as plt

# Custom Scripts:
import common.data.labels.app.label_utils as lu

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

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
])

@app.callback(Output("body-image", "src"),
             [Input('button', 'value')])
def update_body_image(hover_data):
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