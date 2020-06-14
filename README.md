# GapWatch: Making Rural Roads Safer

Unfortunately many road accidents happen in rural areas which could have been avoided if there was more information available to the drivers involved. GapWatch is a project investigating the use of computer vision and audio to increase the amount of information available to drivers exiting dangerous fields with the aim of reducing accidents.

## Use Case Diagram

![Alt text](./common/assets/GapWatch_Use_Case.svg)

## Current Progress

As of June 2020, several major milestones have been achieved:

* Obtained 4 weeks of video data from rural roads in Ireland
* Built a labeling tool for labeling clips of video quickly and accurately
* Labeled videos for "Danger" or "No Danger"
* Verified that a simple Logistic Regression model could achieve performance on unseen video clips and locations comparable to the data it was trained on
* Deployed the simple Logistic Regression model in the field on a laptop for live inference, using  an Arduino and LED

### Data Capturing

![Alt text](./common/assets/GapWatch_footage_box.png)

### Labeling Application

[SHOW THE LABEL APP, PERHAPS A GIF]

### Live Demonstration

The end result is shown in the following video:

[![Watch the video](https://img.youtube.com/vi/I62uLfGEN3U/hqdefault.jpg)](https://youtu.be/I62uLfGEN3U)



## Improvements and Future Plans

There are several major caveats with the current system:

1. The current model was deployed on a laptop with different camera and microphone than the system that was used to capture the training data
2. The logistic regression model does not account for interactions between the audio and image features