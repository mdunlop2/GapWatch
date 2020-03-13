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
    temp_file_name = os.path.join(Path(__file__).absolute().parent, "temp.wav")
    ffmpeg_cmd = ["ffmpeg", "-ss", str(start_seconds),
                    "-i", video_url, "-to", str(current_time),
                    "-ab", "160k",
                    "-ac", "2", "-ar", "44100", "-vn",
                    temp_file_name]
    # We want to continually rewrite the same temporary file
    p = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,
                         stdin=subprocess.PIPE)
    # Will be asked if we want to over-write the existing
    p.stdin.write(b'y')
    # now load into librosa and return the librosa object
    audio, sample_rate = librosa.load(temp_file_name, res_type='kaiser_fast')
    return audio, sample_rate

if __name__ == "__main__":
    # initiate the parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("VIDEO_URL", help="locations of .mp4  file")
    parser.add_argument("FRAME_RATE", help="Video framerate")
    parser.add_argument("FRAME", help="current frame of the video file")

    args = parser.parse_args()
    audio, sample_rate = extract_audio( args.VIDEO_URL,
                                        int(float(args.FRAME)),
                                        int(float(args.FRAME_RATE)),
                                        trail = 5)
    print("Successfully loaded audio. Sample Rate: {}".format(sample_rate))
