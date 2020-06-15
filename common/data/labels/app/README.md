# Image Labeling Tool

This is a simple tool for labeling video frames quickly and efficeintly.

## Live Demonstration for GapWatch

[![Watch the video](https://img.youtube.com/vi/h3VIXlaFNT0/hqdefault.jpg)](https://youtu.be/h3VIXlaFNT0)

Run the following from the project root directory to start the application.

```
python common/data/labels/app/app.py /home/matthew/Documents/video_sample/100_test --PLAYBACK_RATE 5
```

First (and required) option is the directory containing the mp4 videos. Here, I have stored my videos in `/home/matthew/Documents/video_sample/100_test`

Since the aim of this project is to label individual frames and provide inference on live frames from a camera, it is crucial to have some labeled frames to train models and investigate their robustness to unseen videos.


## Design Criteria

* Uses configuration files to generalise to any project
   * Since no such tool exists, I want this to be useful for other people with similar projects when performing live inference
   * This reduces technical debt and allows components to be re used later on in the project when the training pipelines are set up
* Must be storage efficient
   * I had 300GB of mp4 files but in the future this may scale to tens of TB, so individual frames cannot be stored as this would cause an exponential increase in storage space
   * In the future, the project (if funded) will allow for multiple users to label the same frame to allow for majority voting which will further increase the robustness of the data. This would cause massive issues if each frame and label pair were stored individually as in the TFRecord format.
   * OpenCV allows for very efficient extraction of frames from mp4 files, therefore only have to store the frame labels which can then be referred to when classifying a frame
* Must be quick to use (playback at over 1x speed)
   * Labeling costs can be very high for machine learning projects as it requires human time. Watching even just 4 weeks of footage could cost an enormous amount. Instead, use the video player to playback the video at a faster rate.
* Must be intuitive and easy to use
   * Minimum number of clicks possible, reduce probability of mistakes and problems
* Must be simple to use inside the project
   * Ideally not have any external dependencies outside of standard PyPi or Conda Python libraries to reduce project complexity and maintenance

## Future Improvements

### Web App Framework

Plotly Dash has proven to be totally stable and safe for building simple web applications on top of. This application is clearly beyond what Dash was designed to do as a simple dashboarding application.

The current implementation of configuration files is fine for a single user writing labels however will need dramatic changes if multiple label authors are to be allowed. In particular, some sort of cookie storage and retrieval system will need to be constructed in order to allow the author to start where they left off.

Django seems like a great candidate for this system and it greatly expands the freedom to build Web Apps, since the callback system of Plotly Dash has some critical limitations (Outputs can only have one unique Input etc).

Going forward, if this project is funded and multiple author labeling is desired, then a complete rewrite of this system will be necessary. Its possible that the WebApp would be best suited to operate from the cloud to allow for centralised storage of the data.

### Storage

SQLite may not be the best SQL storage solution if the project is migrated to the cloud, will need to investigate other solutions such as Amazon Relational Database.