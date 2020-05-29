'''
This script will create a Tensorflow TFRecord to store
the extracted class from the MobileNetv2 model when
fed batches of frames from video clips

    # eg
    # VIDEO_URL   : /home/matthew/Documents/GapWatch_Videos/0.mp4
    # NUM_FRAMES  : 30
    # FRAME_START : 0
    # FRAME_END   : 1000
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

# video tools
import cv2

# Add the git root directory to python path
sys.path.insert(0,os.getcwd())
app_file_parent_path = Path(__file__).absolute().parent

# Model Related Imports
import keras
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input, decode_predictions
from keras.preprocessing.image import img_to_array
from keras.models import Model



# Custom Scripts:
import common.data.labels.app.label_utils as lu
import common.data.labels.generate_index as gi
import common.data.labels.frame_label_utils as flu
import common.data.labels.frame_sqlite_utils as squ
from common.data.labels.app.config_utils import JSONPropertiesFile

print("Loading Complete")

# write function to extract frames from a video

def video_to_frames(video_url,
                    frame_start,
                    frame_end,
                    num_frames,
                    target_size = (224,224)):
    '''
    Read a .mp4 video file and return a vstack numpy
    array of the images with the correct size
    INPUTS:
    video_url   : String
    num_frames  : Integer
    target_size : Tuple

    OUTPUT:
    array       : (num_frames, target_size[0], target_size[1], 3)
    '''
    # open the video
    video = cv2.VideoCapture(video_url)
    # get framerate
    FRAMERATE = int(video.get(cv2.CAP_PROP_FPS))
    # storage for output vector
    images = np.zeros((num_frames, target_size[0], target_size[1], 3))
    # generate the index of frames to sample
    # we wish to sample uniformly across the clip
    idx_array = np.round(np.linspace(frame_start+1, frame_end-1, num_frames, endpoint = True)).astype("int")
    start = time.time()
    print("Reading {} .mp4 file and \nextracting {} frames between frame {} and {}".format(video_url, num_frames, frame_start, frame_end))
    with progressbar.ProgressBar(max_value=num_frames) as bar:
        for idx in range(num_frames):
            # get the frame number
            frame_number = idx_array[idx]
            # get the frame
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            res, image = video.read()
            # resize to desired dimension
            try:
                image = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)
            except:
                print("Unable to find image in frame {}! Trying next frame".format(frame_number))
                try:
                    video.set(cv2.CAP_PROP_POS_FRAMES, frame_number+1)
                    res, image = video.read()
                    image = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)
                except:
                    print("Unable to find image in frame {}. \n Trying previous frame".format(frame_number +1))
                    try:
                        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number+1)
                        res, image = video.read()
                        image = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)
                    except:
                        print("Unable to find image.")
                        break
            # convert to numpy
            image = img_to_array(image)
            # expand dims
            image = np.expand_dims(image, axis=0)
            # preprocess_input for image_net
            image = preprocess_input(image)
            images[idx,:,:,:] = image
            bar.update(idx)
    print("{} Frames written in {:.3f} seconds".format(num_frames, time.time()-start))
    return images, idx_array, FRAMERATE

def batch_class_inference(model,
                          image_batch):
    '''
    Perform inference on a batch of images of the correct format
    INPUTS:
    model        : Keras Model
    image_batch  : Vstack of preprocesed images

    OUTPUT:
    predicted classes for each image

    NOTES:
    [12 Mar 2020] Mobilenet Max batch size 256, run out of RAM
                (8.76GB used before script start, 16GB total, 67MB swap used)
    '''
    start = time.time()
    preds = model.predict(image_batch, batch_size=image_batch.shape[0])
    print("{} images successfully labelled in {} seconds".format(image_batch.shape[0], time.time()-start))
    return preds


if __name__ == "__main__":
    # initiate the parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("VIDEO_URL", help="locations of .mp4  file")
    parser.add_argument("NUM_FRAMES", help="Number of frames to sample uniformly")
    parser.add_argument("FRAME_START", help="First frame to sample from .mp4  file")
    parser.add_argument("FRAME_END", help="First frame to sample from .mp4  file")

    # Demo: Use Mobilenet
    model = MobileNet(weights='imagenet', include_top=True)

    # read arguments from the command line
    args = parser.parse_args()


    image_batch = video_to_frames(
                        args.VIDEO_URL,
                        int(float(args.FRAME_START)),
                        int(float(args.FRAME_END)),
                        int(args.NUM_FRAMES),
                        target_size = (224,224))
    preds = batch_class_inference(model, image_batch)
    print("Prediction Shape: {}".format(preds.shape))