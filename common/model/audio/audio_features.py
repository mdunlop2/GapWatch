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
import librosa, wave, pyaudio
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
    # check if we have endpoint of the video
    
    start_seconds = max(0, current_time - trail)
    if (start_seconds == 0) or (current_time-start_seconds < trail):
        print("Start: {} End: {} \nAudio Clip too short - increasing to {} seconds".format(start_seconds, current_time, trail))
        current_time = start_seconds + 2 # prevent 1 second tracks appearing
        altered = True
    else:
        altered = False
    print("Start: {} seconds \nEnd: {} seconds".format(start_seconds, current_time))
    temp_file_name = os.path.join(Path(__file__).absolute().parent, "temp.wav")
    ffmpeg_cmd = ["ffmpeg",
                    "-hide_banner","-loglevel", "warning",
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
    # check if it was altered and adjust the array as necessary
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

def video_to_audio(video_url,
                   frame_start,
                   frame_end,
                   num_frames,
                   vid_FPS,
                   m):
    '''
    Read a .mp4 video file and return a vstack numpy
    array of the audio in a format librosa can understand

    Frames are sampled uniformly between frame_start and frame_end
    INPUTS:
    video_url   : String
    frame_start : Starting Frame Number
    frame_end   : Ending Frame Number
                    - if False:
                        We use the last frame in the video
    num_frames  : Integer

    OUTPUT:
    List       : Shape: (num_frames)
                  A list is used since this will be supplied to a starmap
                  for parallelism.
    '''
    temp_fn = "audio.wav" # temporary filename
    # make sure it doesn't exist already
    while os.path.isfile(temp_fn):
        # delete temporary file
        # file deletions appear to fail sometimes
        os.remove(temp_fn)
        time.sleep(0.01)
    
    command = ["ffmpeg", "-i", video_url, "-ab", "160k",
               "-ac", "2", "-ar", "44100", "-vn", temp_fn]
    subprocess.call(command)
    # read the audio file
    wf = wave.open(temp_fn, 'rb')
    p = pyaudio.PyAudio()
    # open stream based on the wave object which has been input.
    RATE = wf.getframerate()
    CHUNK = RATE*m.const_trail()
    stream = p.open(format =
                p.get_format_from_width(wf.getsampwidth()),
                channels = wf.getnchannels(),
                rate = RATE,
                output = True)
    # find the desired frames
    idx_array = m.frame_selection(frame_start, frame_end, num_frames)
    # find corresponding audio clips
    ret = []
    print("Reading {} .mp4 file and \nextracting {} audio clips between frame {} and {}".format(video_url, num_frames, frame_start, frame_end))
    with progressbar.ProgressBar(max_value=num_frames) as bar:
        for idx in range(num_frames):
            # read the audio leading up to frame i
            # this prevents lookahead bias
            # (although in deployment, audio and image features are found sequentially currently)
            audio_pos = max(0,min(wf.getnframes()-1,int(np.floor(RATE*(idx_array[idx]/vid_FPS)-CHUNK))))
            wf.setpos(audio_pos)
            ret.append(np.frombuffer(wf.readframes(int(np.floor(CHUNK))), dtype=np.int16).astype(float))
            bar.update(idx)
    return ret, RATE

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
