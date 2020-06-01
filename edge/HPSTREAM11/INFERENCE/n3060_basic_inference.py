'''
Perform inference using the HP Stream and the basic logistic regression model
which was trained in common/model/training/n3060_basic/n3060_basic.ipynb
'''


# imports
import numpy as np
import matplotlib.pyplot as plt
import joblib, cv2, pyaudio, time, librosa, argparse, os, sys
from pathlib import Path

# custom image extractors
# Add the git root directory to python path
sys.path.insert(0,os.getcwd())
app_file_parent_path = Path(__file__).absolute().parent
from common.model.image.extraction.basic_image_features import basic_image_features


def soundplot(stream):
        t1 = time.time()
        data = np.fromstring(stream.read(CHUNK),dtype=np.int16)
        # process and obtain results
        res = np.mean(librosa.feature.mfcc(y=data.astype(float),
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



if __name__=="__main__":
    # parse any input arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("plot", default=False,
                        help="Enable to view the demo plots as windows, default is False")
    parser.add_argument("t", default=0.5,
                        help="The threshold for which to classify the image, default is 0.5")

    args = parser.parse_args()

    # define hardware specific parameters
    RATE = 44100           # audio sampling rate
    FPS = 60               # initial target audio FPS
    CHUNK = int(RATE/FPS)  # how many samples to listen for each time prediction attempted
    n_mfcc = 10            # number of MFCC components
    n_sec = 5              # number of seconds to record test audio

    ## initialise camera
    cap = cv2.VideoCapture(0)
    # perform camera test
    ret, frame = cap.read()
    print("The camera was initialised: {}".format(ret))
    if args.plot:
        plt.imshow( frame)
        plt.savefig("video_demo.pdf")

    # initialise microphone
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)
    # perform audio test
    print("Testing audio")
    res = np.zeros((n_sec*FPS, n_mfcc))

    for i in range(n_sec*FPS):
        res[i,] = soundplot(stream)
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

    # loop until keyboard exception
    frames = 0
    while True:
        try:
            # run the loop
            # gather camera frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to obtain camera data")
                break
            # resize image to model format
            frame = cv2.resize(frame, (224,224), interpolation=cv2.INTER_CUBIC)
            print(frame.shape)
            # perform image feature extraction
            image_feats = basic_image_features(np.expand_dims(frame, axis=0))
            print(image_feats)
            # gather audio data
            mfccs = soundplot(stream)
        except KeyboardInterrupt:
            break

    # release camera and microphone
    print("Releasing Camera and Microphone")
    cap.release() # release camera
    stream.stop_stream # release audio
    stream.close
    p.terminate()