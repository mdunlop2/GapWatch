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
            "d_mean",
            "var",
            "d_var",
            "kurt",
            "d_kurt",
            "skew",
            "d_skew",
            "mfcc_0",
            "mfcc_1",
            "mfcc_2",
            "mfcc_3",
            "mfcc_4",
            "mfcc_5",
            "mfcc_6",
            "mfcc_7",
            "mfcc_8",
            "mfcc_9",
            "d_mfcc_0",
            "d_mfcc_1",
            "d_mfcc_2",
            "d_mfcc_3",
            "d_mfcc_4",
            "d_mfcc_5",
            "d_mfcc_6",
            "d_mfcc_7",
            "d_mfcc_8",
            "d_mfcc_9"
            ]

def const_config_defaults():
    return {
            'headers'     : ["NA"],
            'sel_headers' : [True],
            'model_store' : "",
            'm_loc'       : "",
            'n_prev'      : 0  
        }

#### MODEL TRAINING AND INFERENCE FUNCTIONS ####
def frame_selection(frame_start, frame_end, num_frames):
    '''
    Define the way that frames are selected within an interval.
    Generally using uniform selection is fine, however this can be changed
    if some future idea suggests it could be useful
    (eg getting pairs of frames to find the derivative)

    idx_array: indices of all frames required
    frame    : indices of the frame performing inference on
    '''
    F1 = np.round(np.linspace(frame_start+1, frame_end-1, num_frames, endpoint = False)).astype("int")
    frame = F1+1 # find the next frame (note that the endpoint is false)
    # the order of concatenation matters because these will be used when processing the features
    idx_array = np.concatenate((F1,frame))
    print("Using Frames: \n{}".format(idx_array))
    return idx_array, frame

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
    # NEW: Find the differences between the two sets of frames
    # recall that F1 is the previous frame, F2 is the current frame.
    n = int(frame_BW.shape[0]/2) # this should ALWAYS be an integer
    F1 = frame_BW[:n,:,:] # first n images
    F2 = frame_BW[n:,:,:] # last n images
    F_diff = F2-F1        # frame difference
    # vstack back onto the current freame (F2)
    # get the model specific features
    # kurtosis, skewness do not support multiple axes
    # need to reshape them to (num_frames, 224*224, 1)
    tmp_batch = np.reshape(F2, (F2.shape[0],
                                -1))
    d_tmp_batch = np.reshape(F_diff, (F_diff.shape[0],
                                -1))
    means = np.mean(tmp_batch, axis=1)
    d_means = np.mean(d_tmp_batch, axis=1)
    variances = np.var(tmp_batch, axis=1)
    d_variances = np.var(d_tmp_batch, axis=1)
    kurtosises = kurtosis(tmp_batch, axis=1)
    d_kurtosises = kurtosis(d_tmp_batch, axis=1)
    skewnesses = skew(tmp_batch, axis=1)
    d_skewnesses = skew(d_tmp_batch, axis=1)
    # stack horizontally
    frame_feats = np.vstack([means, d_means,
                             variances, d_variances,
                             kurtosises, d_kurtosises,
                             skewnesses, d_skewnesses]).T

    audio_feats = np.array([mfcc_from_data(a,n_mfcc, RATE) for a in audio])
    # NEW: Find the difference between the MFCCs
    d_audio_feats = audio_feats[n:] - audio_feats[:n]
    feats = np.hstack((frame_feats, audio_feats[n:], d_audio_feats))
    return feats