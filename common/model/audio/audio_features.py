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
    current_time = int(np.round(frame/frame_rate)) # current time in seconds
    # check if we have ndpoint of the video
    
    start_seconds = max(0, current_time - trail)
    if (start_seconds == 0) or (current_time-start_seconds < 2):
        print("Start: {} End: {} \nAudio Clip too short - increasing to 2 seconds".format(start_seconds, current_time))
        current_time = start_seconds + 2 # prevent 1 second tracks appearing
    print("Start: {} seconds \nEnd: {} seconds".format(start_seconds, current_time))
    temp_file_name = os.path.join(Path(__file__).absolute().parent, "temp.wav")
    ffmpeg_cmd = ["ffmpeg",
                    "-i", video_url,
                    "-ss", str(start_seconds),
                    "-to", str(current_time),
                    "-ab", "160k",
                    "-ac", "2", "-ar", "41000", "-vn",
                    temp_file_name]
    # We want to continually rewrite the same temporary file
    while os.path.isfile(temp_file_name):
        # delete temporary file
        # file deletions appear to fail sometimes
        os.remove(temp_file_name)
        time.sleep(0.01)
    p = subprocess.call(ffmpeg_cmd)
    # Will be asked if we want to over-write the existing
    # p.stdin.write(b'y')
    # p.wait() # force python to wait until .wav is written before reading
    # time.sleep(1)
    while not os.path.isfile(temp_file_name):
        time.sleep(0.1)
    time.sleep(0.1)
    # now load into librosa and return the librosa object
    audio, sample_rate = librosa.load(temp_file_name, res_type='kaiser_fast')
    # while os.path.isfile(temp_file_name):
    #     # delete temporary file
    #     # file deletions appear to fail sometimes
    #     os.remove(temp_file_name)
    #     time.sleep(0.01)
    return audio, sample_rate

def get_mfccs(video_url,
              frame,
              frame_rate,
              n_mfcc,
              trail = 5):
    '''
    Given a .mp4 file, extract the mfccs from the
    audio track and return them.

    INPUTS:
    video_url   : Url pointing to the video the frame came from
    frame       : Frame that we have labelled
    frame_rate  : Integer, video framerate
    n_mfcc      : Number of MFCC features to output
    trail       : Number of seconds of prior audio to include
    '''
    audio, sample_rate = extract_audio(video_url,
                                        frame,
                                        frame_rate,
                                        trail)
    # extract MFCCs
    mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc).T,axis=0)
    # return the MFCCs
    return mfccs
if __name__ == "__main__":
    # initiate the parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("VIDEO_URL", help="locations of .mp4  file")
    parser.add_argument("FRAME_RATE", help="Video framerate")
    parser.add_argument("FRAME", help="current frame of the video file")
    parser.add_argument("N_MFCC", help="number of MFCC features to extract")

    args = parser.parse_args()
    mfccs = get_mfccs(  args.VIDEO_URL,
                        int(float(args.FRAME)),
                        int(float(args.FRAME_RATE)),
                        int(float(args.N_MFCC)),
                        trail = 5)
    print("MFCCs: {}".format(mfccs))
