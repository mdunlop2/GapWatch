'''
Displays the desired frame from the video
'''

# include standard modules
import argparse
import cv2
import matplotlib.pyplot as plt

# initiate the parser
parser = argparse.ArgumentParser()

# add arguments
parser.add_argument("VIDEO_LOC",
        help="Location of the video file. \nMust give full address")
parser.add_argument("--FRAME",
        help="Name of column used in csv to store the file locations",
        default=1)

# read arguments from the command line
args = parser.parse_args()

FRAME = int(args.FRAME)

cap = cv2.VideoCapture(args.VIDEO_LOC)
totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
if FRAME >= 0 & FRAME <= totalFrames:
    # set frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES,FRAME)

ret, frame = cap.read()

print(frame.shape)
plt.imshow(frame)
plt.show()