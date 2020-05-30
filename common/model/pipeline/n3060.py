# HP STREAM 11 Inference Pipeline

import pyaudio, librosa, cv2, time, argparse
import numpy as np

if __name__ == "__main__":
    #~~ read input arguments
    # initiate the parser
    parser = argparse.ArgumentParser()
    # add arguments
    parser.add_argument("-RATE",
                help="Rate (Hz) of the audio",
                default=44100)
    parser.add_argument("-AUDIO_FPS",
                help="Frames per second to sample audio",
                default=30)
    args = parser.parse_args()

    # Audio argument processing
    RATE = int(float(args.RATE))
    AUDIO_FPS = int(float(args.AUDIO_FPS))
    CHUNK = int(RATE/AUDIO_FPS)
    
    #~~ begin the pipeline

    # establish audio connection
    p = pyaudio.PyAudio
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    # establish the camera connection
    cap = cv2.VideoCapture(0)

    #~~ enter the while loop
    while(True):
        try:
            #~ gather the raw data

            #~ process it
            # update long term mean and variance
            
            # standardisation of current values

            #~ perform inference

            #~ relay the results
        except KeyboardInterrupt:
            print("Stopping Loop and closing camera, \nmicrophone connections")
            # safely close the microphone and camera connection
            stream.top_stream
            stream.close
            p.terminate()
            cap.release()


