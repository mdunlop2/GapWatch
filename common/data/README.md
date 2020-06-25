## Data
This directory contains the labeling tool used to label the videos for GapWatch.

### Setup Guide:

If you plan on setting up this project for your own class detection problem, please note that major changes will need to be made when building models if you have more than two classes. The labelling setup is capable of supporting more than two classes but all the code I've used for building models assumes only two classes.

Before you run anything, please open `common/data/labels/app/app.py` in whatever editor you prefer.

In the future I'd like to have the setup process generalised, however if this happens pretty much all of the label app will be ported to Django and so there is no point in implementing a fancy setup script.

If you need to change the class labels, simply find the list called `labels` and change to your classes. These are the classes which will be options for the labeling tool.

You may also wish to change the labels for the SQL table, this is flexible and they are stored as variables at startup also within the `### TEMPORARY FIXES ###` area of the file. Implementing a config file is a bit of a catch22 because there needs to be a configuration to create the configuration file!

Run our labeling tool from the project root directory (for me this is /home/mdunlop/Documents/GapWatch/GapWatch)

```
python common/data/labels/app/app.py --PLAYBACK_RATE 5 "/home/mdunlop/Documents/GapWatch/videos/GapWatch_Videos"
```

`STATIC` is the directory where the videos are stored. The videos may be contained in sub-directories of this directory, that's ok as we find them with a recursive search.

`--PLAYBACK_RATE` represents how fast the playback should be. 7x means that each video will play at 7x speed. I recommend experimenting with this to find what the maximum speed your storage medium can support. I had a 5400rpm Toshiba Eco Hard Disk Drive which could support 4x maximum before the video player would skip frames.

Open up the label application like here:

```
http://127.0.0.1:8050/
```

