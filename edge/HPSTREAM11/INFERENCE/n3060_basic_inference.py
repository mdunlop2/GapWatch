'''
Perform inference using the HP Stream and the basic logistic regression model
which was trained in common/model/training/n3060_basic/n3060_basic.ipynb
'''
# disable warnings
import warnings
warnings.filterwarnings("ignore")

# imports
import numpy as np
import matplotlib.pyplot as plt
import joblib, cv2, pyaudio, time, librosa, argparse, os, sys, logging
from pathlib import Path

# arduino
import pyfirmata
board = pyfirmata.Arduino('/dev/ttyACM0')

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
    # setup logging
    logging.basicConfig(filename='n3060.log', level=logging.DEBUG)
    # parse any input arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("-plot", default=False,
                        help="Enable to view the demo plots as windows, default is False")
    parser.add_argument("-t", default=0.5,
                        help="The threshold for which to classify the image, default is 0.5")

    args = parser.parse_args()

    # define hardware specific parameters
    RATE = 44100           # audio sampling rate
    FPS = 60               # initial target audio FPS
    CHUNK = int(RATE/FPS)  # how many samples to listen for each time prediction attempted
    n_mfcc = 10            # number of MFCC components
    n_sec = 1              # number of seconds to record test audio

    # model specific parameters
    l = -np.log(0.5)/60 # time weighting. 1 minute = 50% weight
    t = float(args.t)

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
            # perform image feature extraction
            image_feats = basic_image_features(np.expand_dims(frame, axis=0))
            # print("img feats shape: ",image_feats.shape)
            # gather audio data
            mfccs = soundplot(stream)
            # print("mfccs     shape: ",mfccs.shape)
            # join the features together in the correct format
            X = np.concatenate((np.squeeze(image_feats), mfccs)).reshape(1,-1)
            # print("X         shape: ",X.shape)
            # perform moving average standardisation
            # if this is the first frame, set mean, variance to the first frame (arbitrarily)
            # since zero varaince would trigger errors (and not be very good for predictions)
            if frames==0:
                ema_mean = X
                # this variance estimate is wrong but this first frame should be disregarded 
                ema_var  = np.ones((len(X))).reshape(1,-1) 
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
            print("Danger Probability: {} FPS: {}".format(p[0,1], total_FPS), flush=True)
            logging.info("Danger Probability: {} FPS: {}".format(p[0,1], total_FPS))
            if p[0,1] > t:
                # we have danger!
                board.digital[13].write(1)
            else:
                board.digital[13].write(0)
        except KeyboardInterrupt:
            break

    # release camera and microphone
    print("Releasing Camera and Microphone")
    cap.release() # release camera
    stream.stop_stream # release audio
    stream.close
    p.terminate()