'''
This is a Keras-style model configuration file which
will allow for more generalised data pipeline to reduce
technical debt when experimenting with model configurations.

Each model must contain at least the following functions (with the same names):
- preprocess_input
- frame_selection

n3060_dif:
An extremely simple model
- 10 MFCCs
- Image features at (224,224) resolution in Black and White (so only one colour channel)
- Standardised using the global mean and standard deviation of the training data
- Difference between this frame image and the previous frame image
- Difference in this frames mfcc and the previous frame mfcc

Why include differences?
The image and audio processes in the field are dependent on numerous factors (wind, time of day, weather)
and so the value of the features generated at any one time are not actuallystationary processes,
there are variations but the processes are the cumulative sum of these variations.
In time-series analysis, to deal with such processes one needs to apply a differential,
where by we use the rate of change of the features as inputs to the model.

To achieve this, the model needs to have not only the most recent frame available, but also
the previous frame. It then finds the difference between these and proceeds as with n3060_std in
feature extraction.
'''

import cv2, librosa, os
from multiprocessing import Pool
from itertools import repeat
import numpy as np
from scipy.stats import skew, kurtosis

#### MODEL SPECIFIC CONSTANTS ####
# ideally these will be stored in a configuration file
def const_n_mfcc():
    return 10

def const_interpol():
    return cv2.INTER_NEAREST

def const_target_size():
    return (224,224)

def const_trail():
    # length of audio track to record during inference (seconds)
    return 1/60

def const_header():
    '''
    Headers to be used throughout training and inference.
    Make sure that the headers line up with the preprocess_input function output
    The first three items in this list will represent:
    - frame label
    - video location
    - frame number
    The rest should describe each feature output by preprocess_input
    '''
    return ["label",
            "video_url",
            "frame",
            "mean",
            "var",
            "kurt",
            "skew",
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

#### MODEL TRAINING AND INFERENCE FUNCTIONS ####
def frame_selection(frame_start, frame_end, num_frames):
    '''
    Define the way that frames are selected within an interval.
    Generally using uniform selection is fine, however this can be changed
    if some future idea suggests it could be useful
    (eg getting pairs of frames to find the derivative)
    '''
    F1 = np.round(np.linspace(frame_start+1, frame_end-1, num_frames, endpoint = False)).astype("int")
    F2 = F1+1 # find the next frame (note that the endpoint is false)
    # the order of concatenation matters because these will be used when processing the features
    ret = np.concatenate((F1,F2))
    print("Using Frames: \n{}".format(ret))
    return ret

def frame_transform(frame):
    '''
    Perform any opencv transformations to the data that may be necessary
    '''
    return frame

def mfcc_from_data(data, n_mfcc, RATE):
    '''
    Extract the mfcc features from a sample of audio data
    '''
    return np.mean(librosa.feature.mfcc(y=data.astype(float),
                                        sr=RATE,
                                        n_mfcc=n_mfcc).T, axis=0)

def preprocess_input(frame, audio,
                    inference = False,
                    RATE = 44100):
    '''
    Preprocessing function, given a frame and audio track
    in the same format that the inference pipeline will be supplied.

    INPUTS:
    frame     : OpenCV cap.read()[1]
                If training the model:
                    This can be treated as a numpy array with dimensions:
                    [2n, frame_width (pixels), frame_height (pixels), 3]
                    Which allows for much faster matrix multiplication in batches of n frames
                    Note that it is assumed that the frame has already been resized to the desired
                    shape.
                If in live inference mode:
                    This can be treated as a numpy array with dimensions:
                    [2, frame_width (pixels), frame_height (pixels), 3]
                    NOTE that this requires np.expand_dims(frame, axis=0) in the inference code,
                    however since this affects all models this is fine.
    audio     : pyaudio.PyAudio data as a numpy array
                If training:
                    List of np.frombuffer(wave.open(file).setpos().readframes(CHUNK)) objects
                    can then perform mfcc inference in parallel with starmap. The training script
                    will need to deal with findding the chunk corresponding to each frame.
                    Supply as list with dim:
                    [2n]
                If in live inference mode:
                    List of length 1 containing a np.fromstring(stream.read(CHUNK)) object
    inference : True if in live inference mode and False otherwise
    RATE      : Audio sample rate

    OUTPUTS:
    features  : The complete features of audio and video
    '''
    # MODEL SPECIFIC PARAMETERS
    n_mfcc = const_n_mfcc()   # possibly replace this with a config file when YAML scripting is setup
    # convert to black and white
    frame_BW = np.array([cv2.cvtColor(np.squeeze(frame[i,:,:,:]).astype(np.float32), cv2.COLOR_BGR2GRAY) for i in range(frame.shape[0])])
    print("frame_BW Shape: {}".format(frame_BW.shape))
    # NEW: Find the differences between the two sets of frames
    # recall that F1 is the previous frame, F2 is the current frame.
    n = int(frame_BW.shape[0]/2) # this should ALWAYS be an integer
    F1 = frame_BW[:n,:,:] # first n images
    F2 = frame_BW[n:,:,:] # last n images
    F_diff = F2-F1        # frame difference
    # get the model specific features
    means = np.mean(F_diff, axis=(1,2))
    variances = np.var(F_diff, axis=(1,2))
    # kurtosis, skewness do not support multiple axes
    # need to reshape them to (num_frames, 224*224, 3)
    tmp_batch = np.reshape(F_diff, (frame.shape[0],
                                  -1))
    kurtosises = kurtosis(tmp_batch, axis=1)
    skewnesses = skew(tmp_batch, axis=1)
    # stack horizontally
    frame_feats = np.vstack([means, variances, kurtosises, skewnesses]).T

    # Currently there is a bug with librosa whereby multiprocessing cannot
    # be used as some objects are shared between the threads.
    # However, fortunately the mfccs can be calculated fast and in somewhat parallel
    # with numba and MKL so this doesn't affect performance too much.
    # Overall, the vast majority of the time for this script to complete is
    # occupied reading and finding the mp4 frames which is limited by hard drive speed.
    # Audio track
    # use starmap to iterate
    # with Pool(1) as pool:
    #     audio_feats = pool.starmap(mfcc_from_data,
    #                                 zip(audio,
    #                                     repeat(n_mfcc),
    #                                     repeat(RATE)))
    audio_feats = np.array([mfcc_from_data(a,n_mfcc, RATE) for a in audio])
    # need to convert to numpy array and then concatenate to the image features
    # combine to find the total features

    # NEW: Find the difference between the two sets of audio features.
    feats = np.hstack((frame_feats, audio_feats))
    return feats
    