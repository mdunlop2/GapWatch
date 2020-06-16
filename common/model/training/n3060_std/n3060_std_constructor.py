'''
General function for building the training data set given a model.
'''
# include standard modules
import importlib, argparse, os.path, os, sys, time, sqlite3, subprocess, progressbar
import pyaudio, librosa, wave

from pathlib import Path
import numpy as np

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

def video_to_audio(video_url,
                   frame_start,
                   frame_end,
                   num_frames,
                   vid_FPS,
                   m):
    '''
    Read a .mp4 video file and return a vstack numpy
    array of the audio in a format librosa can understand

    Frames are sampled uniformly between frame_start and frame_end
    INPUTS:
    video_url   : String
    frame_start : Starting Frame Number
    frame_end   : Ending Frame Number
                    - if False:
                        We use the last frame in the video
    num_frames  : Integer

    OUTPUT:
    List       : Shape: (num_frames)
                  A list is used since this will be supplied to a starmap
                  for parallelism.
    '''
    temp_fn = "audio.wav" # temporary filename
    # make sure it doesn't exist already
    while os.path.isfile(temp_fn):
        # delete temporary file
        # file deletions appear to fail sometimes
        os.remove(temp_fn)
        time.sleep(0.01)
    
    command = ["ffmpeg", "-i", video_url, "-ab", "160k",
               "-ac", "2", "-ar", "44100", "-vn", temp_fn]
    subprocess.call(command)
    # read the audio file
    wf = wave.open(temp_fn, 'rb')
    p = pyaudio.PyAudio()
    # open stream based on the wave object which has been input.
    RATE = wf.getframerate()
    CHUNK = RATE*m.const_trail()
    stream = p.open(format =
                p.get_format_from_width(wf.getsampwidth()),
                channels = wf.getnchannels(),
                rate = RATE,
                output = True)
    # find the desired frames
    idx_array = m.frame_selection(frame_start, frame_end, num_frames)
    # find corresponding audio clips
    ret = []
    print("Reading {} .mp4 file and \nextracting {} audio clips between frame {} and {}".format(video_url, num_frames, frame_start, frame_end))
    with progressbar.ProgressBar(max_value=num_frames) as bar:
        for idx in range(num_frames):
            # read the audio leading up to frame i
            # this prevents lookahead bias
            # (although in deployment, audio and image features are found sequentially currently)
            audio_pos = max(0,min(wf.getnframes()-1,int(np.floor(RATE*(idx_array[idx]/vid_FPS)-CHUNK))))
            wf.setpos(audio_pos)
            ret.append(np.frombuffer(wf.readframes(int(np.floor(CHUNK))), dtype=np.int16).astype(float))
            bar.update(idx)
    return ret, RATE



def featureset_construct(DATABASE,
                         NUM_SAMPLES,
                         STORE,
                         n_mfcc,
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
    print("Label Data Shape: {}".format(label_data.shape))
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
    for i in range(5): # temporarily only take a small sample of vids
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
        audio_batch, RATE = video_to_audio(
                                        label_data[i,0],
                                        int(label_data[i,1]),
                                        int(label_data[i,2]),
                                        int(batch_ref[label]),
                                        frame_rate,
                                        m)

        # get features using model preprocessing stage
        features, headers = m.preprocess_input(frame = image_batch,
                                               audio = audio_batch,
                                               inference = False,
                                               RATE = RATE)
        
        # write to csv

        
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

    # tmp: try to generate a dummy index
    print("Dummy frame selection: \n{}".format(m.frame_selection(0, 99, 10)))

    # tmp: print some of the saved functions
    print("n_mfccs: {}".format(m.const_n_mfcc()))

    label_data = featureset_construct(  args.db,
                                        int(float(args.n)),
                                        args.s,
                                        int(float(m.const_n_mfcc())),
                                        int(float(m.const_trail())),
                                        labels
                                        )
    print(label_data.shape)