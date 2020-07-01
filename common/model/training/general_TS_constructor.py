'''
This script is designed to work similar to general_constructor.py but
instead of the output being a row-per-frame, it is now a time series for each
video, which will enable more advanced models to take advantage of temporal features.

Key differences:
- Each row now corresponds to a time series of several thousand frames
  - Since this is a "ragged" structure, it is best to write as a "tall"
    Dataframe, where each row is a frame of the time series.
- Every video is resized first
  - I found that the OpenCV resize operation was extremely slow, which was
    compounded with being forced to use tensorflow normalisation routine.
    Instead, no normalisation is performed to the pixels and the videos are all
    resized together at once utilising OpenCL on GPU.
  - The end result should be much faster processing of the videos when building the training
    dataset, which is important because there will be several million frames
    (previous pipeline operated at around 2-3 FPS which is ok when only taking ~10k frames)

Note that GT650m achieves only 4.4x speed on resize while i7-3630QM achieves 8.4x speed but runs a lot cooler.
Your milage may vary depending on GPU, this particular one is very weak and old and cannot run Cuda h264 acceleration
which would likely be much faster (need a Pascal GPU for this like GTX 1070 etc.)

I choose to run on the GPU as it sits nicely at 54 degrees C compared to 92 degrees C on CPU only.

Example usage:
python common/model/training/general_TS_constructor.py \
    -GPU "True"
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


def extract_features(DATABASE,
                     m,
                     GPU = False,
                     target_size = (224,224)
                        ):
    '''
    INPUTS:
    DATABASE    : .db file created by common/data/labels/app/app.py
    m           : This is the dataset model file, which outlines how the frames will
                   be processed, also gives information on how to name the new frames database
    GPU         : Boolean, whether to use GPU for resizing or not
    target_size : Tuple, pixels of the resized image.

    '''
    print(f"Resizing all videos in {DATABASE} to {target_size}. Use GPU: {GPU}")
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
    # only interested in labels not NaN!
    label_data = label_data[np.isin(label_data[:,3], labels),:]
    # get the unique videos
    unique, counts = np.unique(label_data[:,0], return_counts=True)
    print(f"Unique Videos: \n{unique}")
    # loop over videos
    for i in range(len(unique)):
    #for i in range(5):
        start = time.time()
        video_url = unique[i] # get our video url
        # find the directory where our output video will be stored
        temp_video_url = "video.mp4"
        # make sure it doesn't exist already
        while os.path.isfile(temp_video_url):
            # delete temporary file
            # file deletions appear to fail sometimes
            os.remove(temp_video_url)
            time.sleep(0.01)
        # perform the conversion
        interpol = m.const_interpol(for_ffmpeg=True)
        if GPU:
            # OpenCL acceleration
            # MUST SPECIFY THE FFMPEG INSTALL NOT IN CONDA!!!!
            command = ["/usr/bin/ffmpeg", "-hwaccel", "vaapi",
                        "-i", video_url,
                        "-vf", f"scale={target_size[0]}:{target_size[1]}",
                        "-sws_flags", interpol,
                        temp_video_url]
        else:
            # Fallback to CPU
            command = ["ffmpeg",
                        "-i", video_url,
                        "-vf", f"scale={target_size[0]}:{target_size[1]}",
                        "-sws_flags", interpol,
                        temp_video_url]
        subprocess.call(command)
        # Extract every frame from the image file into a batch
        # we do not perform any normalisation, this will be up to the model to enforce.
        # read the video
        video = cv2.VideoCapture(temp_video_url)
        # find video end
        # NOTE: I started running out of RAM when videos reached ~8000 Frames (required massive arrays, about 100m cells in total)
        # so its probably best to split the videos into smaller batches.
        max_batch_size = 4000 # frames
        tot_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        n_batches = (tot_frames // max_batch_size) +1
        print(f"Number of Batches: {n_batches}")

        for b in range(n_batches):
            # define start and end point
            frame_end = min((b+1)*max_batch_size, tot_frames)
            frame_start = b*max_batch_size
            # get framerate
            frame_rate = int(video.get(cv2.CAP_PROP_FPS))
            # NOTE: It is well documented that OpenCV cap.set(cv2.CAP_PROP_POS_FRAMES) is extremely
            # slow, instead one should use the video.read() which iterates the frames.
            # since we'll be reading all the frames of the video, we just need to pass through it once.
            # Then, the pre-process input function can deal with obtaining the frame-to-frame derivatives
            images = np.zeros((frame_end-frame_start, target_size[0], target_size[1]))
            print("Reading {} .mp4 file and \nextracting frames between frame {} and {}".format(video_url, frame_start, frame_end))
            video.set(cv2.CAP_PROP_POS_FRAMES, 0) # start from beginning
            with progressbar.ProgressBar(max_value=(frame_end-frame_start)) as bar:
                for fr in range(frame_end-frame_start):
                    # get the frame
                    res, image = video.read()
                    # Store the image
                    images[fr,:,:] = np.expand_dims(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), axis=0)
                    bar.update(fr)
            # stack into the desired format for getting frame to frame derivatives
            n_derivatives = m.frame_derivatives()
            img_batch = images[n_derivatives:,:,:] # "current frame"
            frames = np.arange(n_derivatives+frame_start,frame_end) # indices of the current frame to refer to video
            for j in range(n_derivatives):
                # stack on the desired previous frames
                to_add = images[(n_derivatives-j-1):(-j-1),:,:]
                print(f"Vstacking on array of shape {to_add.shape} to image_batch")
                img_batch = np.vstack((img_batch, to_add))
            # obtain the batch of audio data
            audio_batch, frames, RATE = af.video_to_audio_TS(
                                            video_url,
                                            frame_start,
                                            frame_end,
                                            frame_rate,
                                            m)
            # how fast are we getting frame and audio in usable format?
            print(f"Video and audio available at {(frame_end-frame_start)/(time.time()-start)} FPS")
            # ready for batch preprocessing as usual!
            # get features using model preprocessing stage
            print(f"Image batch shape: {img_batch.shape} prev frames: {n_derivatives}")
            print(f"Audio batch shape: {audio_batch.shape} prev frames: {n_derivatives}")
            features = m.preprocess_input(  frame = img_batch,
                                            audio = audio_batch,
                                            inference = False,
                                            RATE = RATE)
            headers = m.const_header()
            # write to csv with pandas
            df = pd.DataFrame(data = features, columns=headers[3:], index=frames)
            # figure out what label to give to each observation.
            df[headers[0]] = np.NaN
            # iterate over each recorded SQL row to label the appropriate frame
            rel_label_data = label_data[label_data[:,0]==video_url,:]
            for k in range(counts[i]):
                frame_label       = rel_label_data[k,3]
                frame_label_start = rel_label_data[k,1].astype(float)
                frame_label_end   = rel_label_data[k,2].astype(float)
                print(f"frame label: {frame_label}\nstart: {frame_label_start}\nframe end: {frame_label_end}")
                # make sure that the correct rows are labelled
                df.loc[(frames >= frame_label_start)&(frames <= frame_label_end), headers[0]] = frame_label
            df[headers[1]] = video_url  # video location
            df[headers[2]] = frames          # frame numbers
            # make sure that correct order of headers is saved to csv
            df = df[headers]
            # append to previous csv if it exists
            df.to_csv(args.s, mode='a', header=not os.path.exists(args.s), index=False)
            del(df, audio_batch, img_batch, images, frames,features,to_add) # memory cleanup
        print(f"Model features available at {(tot_frames)/(time.time()-start)} FPS")

        


if __name__=="__main__":
    # parse any input arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", default="common.model.training.LSTM.LSTM",
                        help="Location of the model file.")
    parser.add_argument("-db", default="common/data/labels/app/frames.db",
                        help='''Database to draw the frame labels from.
                                Needs to be created by the labeling app
                                supplied in this repository
                                (common/data/labels/app/app.py).''')
    parser.add_argument("-s", default = "common/model/training/LSTM/TS.csv" ,
                        help="Location in which to write the output dataset")
    parser.add_argument("-GPU", default=False,
                        help="To use the OpenCL vaapi acceleration. If False, will \
                              fall back to CPU")
    args = parser.parse_args()
    m = importlib.import_module(args.m) # import the model file

    target_size = m.const_target_size()

    extract_features(args.db,
                     m,
                     args.GPU,
                     target_size
                        )

    

