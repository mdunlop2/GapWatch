'''
Demonstrate any number of models on a given video using the general model format and configuration files.

This will allow easy direct comparison of models, currently reading
the mp4 file is the most expensive part of the pipeline so allowing
multiple models to infer as the features are generated from frames
will save a lot of time as number of models increases.

It should be available as a function also such that it can be called from
inside jupyter notebooks to easily compare models
'''

# disable warnings
import warnings
warnings.filterwarnings("ignore")

# imports
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import joblib, subprocess, cv2, pyaudio, wave, time, librosa, argparse, os, sys, logging
import progressbar, importlib
from pathlib import Path

# custom image extractors
# Add the git root directory to python path
sys.path.insert(0,os.getcwd())
app_file_parent_path = Path(__file__).absolute().parent
from common.model.image.extraction.basic_image_features import basic_image_features
# config utilities
from common.data.labels.app.config_utils import JSONPropertiesFile


def audio_numpy(wf, frame_no, vid_FPS, CHUNK, RATE):
    '''
    INPUTS:
    wf       : wave_read object
    frame_no : Current video frame
    vid_FPS  : Current video framerate
    CHUNK    : Amount of audio to read

    OUTPUTS:
    data     : Numpy array that can be used for feature extraction
    '''
    audio_pos = max(0,min(wf.getnframes()-1,int(np.floor(RATE*(frame_no/vid_FPS)-CHUNK))))
    wf.setpos(audio_pos)
    data = np.frombuffer(wf.readframes(CHUNK), dtype=np.int16)
    return np.expand_dims(data, axis=0)

def inference_demo(v, o, m):
    # define hardware specific parameters
    RATE = 44100           # audio sampling rate
    FPS = 60               # initial target audio FPS
    CHUNK = int(RATE/FPS)  # how many samples to listen for each time prediction attempted

    ## initialise camera
    cap = cv2.VideoCapture(v)
    vid_FPS = cap.get(cv2.CAP_PROP_FPS)
    # perform camera test
    ret, frame = cap.read()
    print("The camera was initialised: {}".format(ret))

    # initialise audio track from mp4 file
    temp_fn = "audio.wav" # temporary filename
    # make sure this filename doesn't exist
    while os.path.isfile(temp_fn):
        # delete temporary file
        # file deletions appear to fail sometimes
        os.remove(temp_fn)
        time.sleep(0.01)
    command = ["ffmpeg", "-i", args.v, "-ab", "160k",
               "-ac", "2", "-ar", "44100", "-vn", temp_fn]
    subprocess.call(command)

    # read the audio file
    wf = wave.open(temp_fn, 'rb')
    p = pyaudio.PyAudio()
    # open stream based on the wave object which has been input.
    stream = p.open(format =
                p.get_format_from_width(wf.getsampwidth()),
                channels = wf.getnchannels(),
                rate = wf.getframerate(),
                output = True)
    
    # initialise models
    cfgs = []
    clfs = []
    prev = []

    for i in range(len(m)):
        # load the configuration file
        config_file = JSONPropertiesFile(m[i])
        config = config_file.get()
        cfgs.append(config)
        clfs.append(joblib.load(config["model_store"])) # load the classifiers
        prev.append(config["n_prev"]) # number of previous frames required
    print("Models loaded")

    # all share the same dataset config, s
    s = importlib.import_module(cfgs[0]["m_loc"]) # loads the model location

    # loop until keyboard exception or video complete
    frames = 0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    offset = int(np.ceil(FPS/vid_FPS))+1*int(round(vid_FPS)) # audio may not be read correctly in last second

    # store predicted probability and the features and were supplied to model
    headers = s.const_header()
    res = np.zeros((n_frames-offset, len(headers[3:])+len(m)))
    start = time.time()
    start_frame = 0 # frame to start inference from
    with progressbar.ProgressBar(max_value=n_frames-offset-start_frame) as bar:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for i in range(n_frames-offset-start_frame):
            try:
                # run the loop
                # gather camera frame
                
                ret, frame = cap.read()
                # put frame into correct format
                frame = np.expand_dims(cv2.resize(frame, (224,224), interpolation=cv2.INTER_NEAREST), axis=0)
                if not ret:
                    print("Failed to obtain camera data")
                    break
                # gather audio data
                audio = audio_numpy(wf, i, vid_FPS, CHUNK, RATE)
                # begin storing historical data
                if i==0:
                    # no previous data, instead update with new
                    image_batch = np.repeat(frame, max(prev)+1, axis=0)
                    audio_batch = np.repeat(audio, max(prev)+1, axis=0)
                else:
                    image_batch[1:] = image_batch[:-1] # update all prev, drop last
                    audio_batch[1:] = audio_batch[:-1]
                    image_batch[0]  = frame            # first entry becomes new frame
                    audio_batch[0]  = audio
                # obtain the dataset features
                feats = s.preprocess_input(image_batch, audio_batch,
                                            inference = True,
                                            RATE = RATE)
                # obtain classifications from each model
                p   = np.zeros((len(m)))
                for j in range(len(clfs)):
                    # check if we need a subset of the features only
                    sel_headers = cfgs[j]["sel_headers"] # some array of booleans
                    feats_j = feats[:,sel_headers[3:]] # selected features for the model
                    p[j] = clfs[j].predict_proba(feats_j)[0,1] # save the predicted probability
                res[i,] = np.insert(feats,0,p)
                bar.update(i)
                

            except KeyboardInterrupt:
                break
    effective_FPS = (time.time()-start)/(n_frames-offset)
    print("Effective Frame Rate: {}".format(effective_FPS))
    # write the results to a csv
    df = pd.DataFrame(res)
    df.to_csv(o, header=None)
    # release camera and microphone
    print("Releasing Camera and Microphone")
    cap.release() # release camera
    stream.stop_stream # release audio
    stream.close
    p.terminate()
    return df

if __name__=="__main__":
    # parse any input arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", help='''List of model configuration file locations.
                                                 All of these models should support the same
                                                 dataset configuration
                                                 eg common.model.training.n3060_dif.n3060_dif''',
                        nargs="+", required=True)

    parser.add_argument("-v",
                        help=".mp4 video to perform inference on")

    parser.add_argument("-o",
                        help="Out file location to store the model input and output")

    
    args = parser.parse_args()

    print("m: \n{}".format(args.m))

    df = inference_demo(args.v, args.o, args.m)



    