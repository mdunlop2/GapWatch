'''
Perform Inference on a video to visualise the model performance.
'''

'''
Perform inference using the HP Stream and the basic logistic regression model
which was trained in common/model/training/n3060_basic/n3060_basic.ipynb
'''
# disable warnings
import warnings
warnings.filterwarnings("ignore")

# imports
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import joblib, subprocess, cv2, pyaudio, wave, time, librosa, argparse, os, sys, logging
import progressbar
from pathlib import Path

# custom image extractors
# Add the git root directory to python path
sys.path.insert(0,os.getcwd())
app_file_parent_path = Path(__file__).absolute().parent
from common.model.image.extraction.basic_image_features import basic_image_features


def soundplot(wf, frame_no, vid_FPS, CHUNK):
    '''
    INPUTS:
    wf       : wave_read object
    frame_no : Current video frame
    vid_FPS  : Current video framerate
    CHUNK    : Amount of audio to read 
    '''
    audio_pos = max(0,min(wf.getnframes()-1,int(np.floor(RATE*(frame_no/vid_FPS)-CHUNK))))
    wf.setpos(audio_pos)
    data = wf.readframes(CHUNK)
    # process and obtain results
    res = np.mean(librosa.feature.mfcc(y=np.frombuffer(data, dtype=np.int16).astype(float),
                                        sr=RATE,
                                        n_mfcc=n_mfcc).T, axis=0)
    return res

def update_ema(lam, S, X, t1, t2):
    '''
    INPUTS:
    lambda : Exponential weighting parameter
    S      : Exponentially weighted moving average component at time t1
    X      : Value recorded at time t2

    Times t1, t2 are in seconds
    
    OUTPUTS:
    S      : Exponentially weighted moving average component at time t
    '''
    u = np.exp(-(t2-t1)*lam)
    return u*S + (1-u)*X

def get_std_vals(csv_loc):
    # read in the data and establish the headers
    headers = [
        "label",
        "video_url",
        "mean_0",
        "mean_1",
        "mean_2",
        "var_0",
        "var_1",
        "var_2",
        "kurt_0",
        "kurt_1",
        "kurt_2",
        "skew_0",
        "skew_1",
        "skew_2",
        "mfcc_0",
        "mfcc_1",
        "mfcc_2",
        "mfcc_3",
        "mfcc_4",
        "mfcc_5",
        "mfcc_6",
        "mfcc_7",
        "mfcc_8",
        "mfcc_9"
    ]
    df = pd.read_csv(csv_loc,
                    names=headers)
    mean = np.mean(df.loc[df[headers[0]]=="No_Danger" , headers[2:]])
    var  = np.var( df.loc[df[headers[0]]=="No_Danger" , headers[2:]])
    print("Training Mean: \n {}\nTraining Variance: \n{}".format(mean, var))
    return mean.values.reshape(1,-1), var.values.reshape(1,-1)



if __name__=="__main__":
    # parse any input arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("-v",
                        help=".mp4 video to perform inference on")

    parser.add_argument("-plot", default=False,
                        help="Enable to view the demo plots as windows, default is False")
    parser.add_argument("-t", default=0.5,
                        help="The threshold for which to classify the image, default is 0.5")
    parser.add_argument("-o",
                        help="Out file location to store the model input and output")

    args = parser.parse_args()

    # define hardware specific parameters
    RATE = 44100           # audio sampling rate
    FPS = 60               # initial target audio FPS
    CHUNK = int(RATE/FPS)  # how many samples to listen for each time prediction attempted
    n_mfcc = 10            # number of MFCC components
    n_sec = 1              # number of seconds to record test audio

    ## model specific parameters
    # k: time weighting. k seconds between t1 and t2 gives t2 50% weight
    k = 1
    l = -np.log(0.5)/k
    t = float(args.t)

    ## initialise camera
    cap = cv2.VideoCapture(args.v)
    vid_FPS = cap.get(cv2.CAP_PROP_FPS)
    # perform camera test
    ret, frame = cap.read()
    print("The camera was initialised: {}".format(ret))
    # attempt to get image features
    frame = cv2.resize(frame, (224,224), interpolation=cv2.INTER_CUBIC)
    image_feats = basic_image_features(np.expand_dims(frame, axis=0))

    if args.plot:
        plt.imshow( frame)
        plt.savefig("video_demo.pdf")

    # initialise audio track from mp4 file
    temp_fn = "audio.wav" # temporary filename
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

    # perform audio test
    print("Testing audio")
    res = np.zeros((n_sec*FPS, n_mfcc))

    for i in range(n_sec*FPS):
        res[i,] = soundplot(wf, i, vid_FPS, CHUNK)
    # plot the results
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(1,1,1)
    for i in range(n_mfcc):
        ax.plot(range(n_sec*FPS), (res[:,i]-np.mean(res[:,i]))/np.std(res[:,i]))
    if args.plot:
        plt.savefig("audio_demo.pdf")
        plt.show()
    

    # initialise model
    clf = joblib.load("common/model/training/n3060_basic/n3060_basic.pkl")

    # loop until keyboard exception or video complete
    frames = 0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    offset = int(np.ceil(FPS/vid_FPS))+1 # frames to offset audio by

    # store everything that happens
    res = np.zeros((n_frames-offset, n_mfcc+image_feats.shape[1]+1))
    # store predicted probability and the features and were supplied to model
    with progressbar.ProgressBar(max_value=n_frames) as bar:
        for i in range(n_frames-offset):
            try:
                # run the loop
                # gather camera frame
                ret, frame = cap.read()
                if not ret:
                    print("Failed to obtain camera data")
                    break
                # resize image to model format
                frame = cv2.resize(frame, (224,224), interpolation=cv2.INTER_CUBIC)
                # perform image feature extraction
                image_feats = basic_image_features(np.expand_dims(frame, axis=0))
                # print("img feats shape: ",image_feats.shape)
                # gather audio data
                mfccs = soundplot(wf, i, vid_FPS, CHUNK)
                # print("mfccs     shape: ",mfccs.shape)
                # join the features together in the correct format
                X = np.concatenate((np.squeeze(image_feats), mfccs)).reshape(1,-1)
                # print("X         shape: ",X.shape)
                # perform moving average standardisation
                # if this is the first frame, set mean, variance to the first frame (arbitrarily)
                # since zero varaince would trigger errors (and not be very good for predictions)
                if frames==0:
                    # ema_mean = X
                    # # this variance estimate is wrong but this first frame should be disregarded 
                    # ema_var  = np.ones((len(X))).reshape(1,-1)
                    # extract mean and variance from the training data to get a good first guess
                    ema_mean, ema_var = get_std_vals("/home/matthew/Documents/GapWatch/common/model/training/n3060_basic/basic_image_mfcc.csv")
                    frame_time = time.time()
                    total_FPS = np.NaN
                    frames+=1
                else:
                    ema_mean = update_ema(l, ema_mean, X, frame_time, time.time())
                    ema_var  = update_ema(l, ema_var  , (X-ema_mean)**2, frame_time, time.time())
                    total_FPS = 1/(time.time()-frame_time)
                    frame_time = time.time()
                    frames+=1
                # standardise data
                X_std = (X-ema_mean)/np.sqrt(ema_var)
                # perform inference
                p = clf.predict_proba(X_std)
                # print("Danger Probability: {} FPS: {}".format(p[0,1], total_FPS), flush=True)
                # store results
                res[i,] = np.insert(X_std,0,p[0,1])
                bar.update(i)

            except KeyboardInterrupt:
                break
    # write the results to a csv
    pd.DataFrame(res).to_csv(args.o, header=None)
    # release camera and microphone
    print("Releasing Camera and Microphone")
    cap.release() # release camera
    stream.stop_stream # release audio
    stream.close
    p.terminate()