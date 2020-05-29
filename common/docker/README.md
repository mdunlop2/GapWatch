# Docker
Detailed instructions on how to set up docker environments.

## Intel Celeron N3060 Base Image

It turns out that it is Docker is not designed with sharing WebCams or Microphones with containers in mind. Instead it is better to simply work with a single virtual environment for the machine, where I have successfully tested the camera with OpenCV.

Celeron N3060 does **NOT** support AVX instructions, so tensorflow >1.5 won't run (can compile from source but this can take up to 25 hours and is not guaranteed to work).

There are two cameras and only two threads so this will effectively be a single threaded pipeline for each camera.

Considering there are only two threads, the potential overhead of transferring image data from opencv to tensorflow and back to numpy would probably mitigate any performance gain from tensorflow in single threaded workloads.