'''
MOBILENET FEATURESET CONSTRUCTOR

The purpose of this file is to create a csv file containing
the desired image features and audio features.

The Machine Learning and Data Science workflows should be applied
directly to the output of this script
'''
# include standard modules
import argparse
import os.path
import os
import sys
import pandas as pd
from pathlib import Path
import time
import numpy as np
import progressbar
import sqlite3

# video tools
import cv2

# Add the git root directory to python path
sys.path.insert(0,os.getcwd())
app_file_parent_path = Path(__file__).absolute().parent

# Custom Scripts:
import common.data.labels.app.label_utils as lu
import common.data.labels.generate_index as gi
import common.data.labels.frame_label_utils as flu
import common.data.labels.frame_sqlite_utils as squ
import common.model.image.extraction.MobileNet_class as mc
import common.model.image.extraction.basic_image_features as bif
import common.model.audio.audio_features as af
from common.data.labels.app.config_utils import JSONPropertiesFile

### TEMPORARY ###
# store some parameters which will be managed by YAML script

labels = [
    'No_Danger',
    'Danger'
]

# Set the column name for FRAME_URLS_FILE and labels database
table_name = "data"

FRAME_COLNAME = "frame"
LABEL_COLNAME = "label"
VIDEO_URL_COLNAME = "video_url"
AUTHOR_COLNAME = "author"
TIMESTAMP_COLNAME = "timestamp"
FRAME_START_COLNAME = "frame_start"
FRAME_END_COLNAME = "frame_end"

### \TEMPORARY ###

def featureset_construct(DATABASE,
                         NUM_SAMPLES,
                         STORE,
                         W,
                         TRAIL,
                         labels
                         ):
    '''
    INPUTS:
    DATABASE    : .db file created by common/data/labels/app/app.py
    NUM_SAMPLES : Total number of samples to include in the dataset (all classes)
    STORE       : Folder in which to store the output dataset
    TRAIL       : Maximum length of audio prior to frame to use (seconds)
    W           : Number of MFCC features to include
    X           : Number of LTP features to include
    Y           : Number of Image Features features to include 
                   (NOT YET IMPLEMENTED!)
    '''
    connex = sqlite3.connect(DATABASE, check_same_thread=False)
    cur = connex.cursor()
    # find out how many samples to allocate to each clip
    # how many videos are of each class?
    label_sql = '''
                SELECT {}, {}, {}, {}
                FROM {}
                WHERE {} = '{}'
                '''.format(
                    VIDEO_URL_COLNAME,
                    FRAME_START_COLNAME,
                    FRAME_END_COLNAME,
                    LABEL_COLNAME,
                    table_name,
                    AUTHOR_COLNAME,
                    'Default'
                )
    # NOTE: Big changes required to this if multiple authors implemented!
    cur.execute(label_sql)
    label_data = np.array(cur.fetchall())
    # NOTE: temporarily take saubset
    label_data = label_data[1:7,:]
    # loop over labels
    # want to find how many clips there are
    # so that we sample uniformly from each and still have
    # balanced number of frames for each class
    unique, counts = np.unique(label_data[:,3], return_counts=True)
    frame_target = np.floor(NUM_SAMPLES/len(labels)) # total frames per class
    batch_size = np.round(frame_target/counts) # batch size for each class
    print("Labels: {} \nCounts: {} \nClip Frames: {}".format(unique, counts, batch_size))
    batch_ref = dict(zip(unique, batch_size))
    # first initiate the keras model
    for i in range(len(label_data[:,0])):
        # get our label
        label = label_data[i,3]
        # now obtain the features for each batch and write to csv file
        image_batch, frames, frame_rate = mc.video_to_frames(
                                        label_data[i,0],
                                        int(label_data[i,1]),
                                        int(label_data[i,2]),
                                        int(batch_ref[label]),
                                        target_size = (224,224))
        image_features = bif.basic_image_features(image_batch)
        # Attempt to extract audio features and write to csv
        for j in range(len(frames)):
            # open csv file to dump features
            csv = open(STORE, "a")
            # put class vectors in csv format
            img_vec = image_features[j]
            img_vec = ",".join([str(v) for v in img_vec])
            # obtain audio features
            print("Frame: {} Second: {}".format(frames[j], frames[j]/frame_rate))
            mfcc_vec  = af.get_mfccs(
                                    label_data[i,0],
                                    frames[j],
                                    frame_rate,
                                    W,
                                    trail = TRAIL)
            mfcc_vec = ",".join([str(m) for m in mfcc_vec])
            csv.write("{},{},{},{}\n".format(label,label_data[i,0],img_vec, mfcc_vec))
            csv.close
    return label_data
    

if __name__ == "__main__":
    # initiate the parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("-DATABASE", help="SQLite Database of labelled videos")
    parser.add_argument("-NUM_SAMPLES", help="Total Dataset Size")
    parser.add_argument("-STORE", help="Folder in which to store the output dataset")
    parser.add_argument("-W", help="Number of MFCC features to include")
    parser.add_argument("-TRAIL", default=5, help="Maximum length of audio prior to frame to use (seconds)")
    # read arguments from the command line
    args = parser.parse_args()
    
    label_data = featureset_construct(  args.DATABASE,
                                        int(float(args.NUM_SAMPLES)),
                                        args.STORE,
                                        int(float(args.W)),
                                        int(float(args.TRAIL)),
                                        labels
                                        )
    print(label_data.shape)