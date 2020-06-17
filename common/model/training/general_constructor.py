'''
General function for building the training data set given a model.
'''
# include standard modules
import importlib, argparse, os.path, os, sys, time, sqlite3, subprocess, progressbar
import pyaudio, librosa, wave

from pathlib import Path
import numpy as np
import pandas as pd

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
import common.model.audio.audio_features as af

### TEMPORARY ###
# store some parameters which will be managed by YAML script on the
# first run of the app.py labeling application.

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
                         n_mfcc,
                         TRAIL,
                         labels,
                         m
                         ):
    '''
    INPUTS:
    DATABASE    : .db file created by common/data/labels/app/app.py
    NUM_SAMPLES : Total number of samples to include in the dataset (all classes)
    STORE       : Folder in which to store the output dataset
    TRAIL       : Maximum length of audio prior to frame to use (seconds)
    W           : Number of MFCC features to include
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
    # loop over labels
    # want to find how many clips there are
    # so that we sample uniformly from each and still have
    # balanced number of frames for each class
    unique, counts = np.unique(label_data[:,3], return_counts=True)
    frame_target = np.floor(NUM_SAMPLES/len(labels)) # total frames per class
    batch_size = np.round(frame_target/counts) # batch size for each class
    print("Labels: {} \nCounts: {} \nClip Frames: {}".format(unique, counts, batch_size))
    batch_ref = dict(zip(unique, batch_size))
    # for i in range(len(label_data[:,0])):
    for i in range(5):
        # get our label
        label = label_data[i,3]
        # obtain the frames in batches
        image_batch, frames, frame_rate = mc.video_to_frames(
                                        label_data[i,0],
                                        int(label_data[i,1]),
                                        int(label_data[i,2]),
                                        int(batch_ref[label]),
                                        target_size = (224,224),
                                        m = m)
        # obtain the batch of audio data
        audio_batch, RATE = af.video_to_audio(
                                        label_data[i,0],
                                        int(label_data[i,1]),
                                        int(label_data[i,2]),
                                        int(batch_ref[label]),
                                        frame_rate,
                                        m)

        # get features using model preprocessing stage
        features = m.preprocess_input(frame = image_batch,
                                               audio = audio_batch,
                                               inference = False,
                                               RATE = RATE)
        
        headers = m.const_header()
        # write to csv with pandas
        df = pd.DataFrame(data = features, columns=headers[3:])
        df[headers[0]] = label           # labels
        df[headers[1]] = label_data[i,0] # video location
        df[headers[2]] = frames          # frame numbers
        # append to previous csv if it exists
        df.to_csv(args.s, mode='a', header=not os.path.exists(args.s))

        
    return label_data


if __name__=="__main__":
    # parse any input arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", default="common.model.training.n3060_std.n3060_std",
                        help="Location of the model file.")
    parser.add_argument("-db", default="common/data/labels/app/frames.db",
                        help='''Database to draw the frame labels from.
                                Needs to be created by the labeling app
                                supplied in this repository
                                (common/data/labels/app/app.py).''')
    parser.add_argument("-n", default=8000, help="Total number of frames to extract from all videos.")
    parser.add_argument("-s", help="Location in which to write the output dataset")

    args = parser.parse_args()

    m = importlib.import_module(args.m) # import the model file

    label_data = featureset_construct(  args.db,
                                        int(float(args.n)),
                                        args.s,
                                        int(float(m.const_n_mfcc())),
                                        int(float(m.const_trail())),
                                        labels,
                                        m
                                        )