'''
Utilities for working with data labels
'''
import cv2
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import base64

def get_image(cap, FRAME):
    '''
    INPUTS
    cap
        - Open CV cv2.VideoCapture Object
    FRAME
        - Integer, 0 <= FRAME <= LAST_FRAME
        - LAST_FRAME is number of frames in the video

    OUTPUT:
    Numpy Array
        - Shape: (1080, 1920, 3) (for 1080p avi video)
    '''
    LAST_FRAME = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if FRAME >= 0 & FRAME <= LAST_FRAME:
        # set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES,FRAME)
    else:
        print("INVALID FRAME NUMBER!")
    
    ret, frame = cap.read()
    return frame

def fig_to_uri(in_fig, close_all=True, **save_args):
    # type: (plt.Figure) -> str
    """
    Save a figure as a URI
    :param in_fig:
    :return:

    Source:
    https://github.com/4QuantOSS/DashIntro/blob/master/notebooks/Tutorial.ipynb
    """
    out_img = BytesIO()
    in_fig.savefig(out_img, format='png', **save_args)
    if close_all:
        in_fig.clf()
        plt.close('all')
    out_img.seek(0)  # rewind file
    encoded = base64.b64encode(out_img.read()).decode("ascii").replace("\n", "")
    return "data:image/png;base64,{}".format(encoded)