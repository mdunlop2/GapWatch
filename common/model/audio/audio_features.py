'''
The purpose of this script is to extract audio features
from .mp4 files between two frames.

We require ffmpeg to extract the audio from the .mp4 file

At the edge, will have a dedicated microphone to retrieve audio records
from instead which may result in a boost in performance
'''
# include standard modules
import argparse
import os.path
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys
import pandas as pd
from pathlib import Path
import time
import numpy as np
import progressbar
import librosa
import subprocess

def extract_audio(video_url,
                  frame,
                  frame_rate,
                  trail = 5):
    '''
    Given a .mp4 file, we create a temporary .wav file
    and load it into librosa, returning it

    INPUTS:
    video_url   : Url pointing to the video the frame came from
    frame       : Frame that we have labelled
    frame_rate  : Integer, video framerate
    trail       : Number of seconds of prior audio to include

    OUTPUTS:
    audio       : librosa.load() object
    '''
    # first work out which segment of audio we are interested in
    current_time = int(np.ceil(frame/frame_rate)) # current time in seconds
    start_seconds = max(0, current_time - trail)
    print("Start: {} seconds \nEnd: {} seconds".format(start_seconds, current_time))
    temp_file_name = "/home/matthew/Documents/test.mp4"
    ffmpeg_cmd_1 = ["ffmpeg", "-ss", str(start_seconds),
                    "-i", video_url, "-to", str(current_time),
                    temp_file_name]
    subprocess.call(ffmpeg_cmd_1)
    # now convert the temporary file to .wav
    ffmpeg_cmd_2 = ["ffmpeg", "-i", temp_file_name, "-ab", "160k",
                    "-ac", "2", "-ar", "44100", "-vn",
                    temp_file_name.replace('mp4', 'wav')]
    subprocess.call(ffmpeg_cmd_2)

if __name__ == "__main__":
    # initiate the parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("VIDEO_URL", help="locations of .mp4  file")
    parser.add_argument("FRAME_RATE", help="Video framerate")
    parser.add_argument("FRAME", help="current frame of the video file")

    args = parser.parse_args()
    extract_audio(args.VIDEO_URL,
                  int(float(args.FRAME)),
                  int(float(args.FRAME_RATE)),
                  trail = 5)
