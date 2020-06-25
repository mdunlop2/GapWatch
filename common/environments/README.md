# Environments
Instructions on how to set up environments.

## Local Development (dev)

This is the environment used to rapidly prototype the image labelling infrastructure and the early SKlearn models. It does require tensorflow and keras since we use the imagenet preprocessing format. The easiest way to set this up is with conda environments.

### Notable Conflicts

1. Librosa and Numba
   * Librosa breaks with numba versions beyond 0.48.0, they are working on a fix, see https://github.com/librosa/librosa/issues/1113

### Step by Step 

Start by installing opencv with opencl capabilities. I had a laptop with both Intel HD Graphics 4000 and nVidia GT650m. GT650m is much better (although still very outdated at this stage) but included both for future reference if necessary.

*(Using Intel HD Graphics 4000)*
```
sudo apt-get install -y --no-install-recommends \
        ocl-icd-libopencl1 \
        clinfo \
        beignet \
        opencl-headers
```

*(Using nVidia GT650m GPU)*

Getting nVidia drivers setup can be tricky and can vary between OS and versions. I use POP! OS 18 which comes with nVidia driver 430.34 and `clinfo` shows that it is available for opencl.

Make sure that the nVidia GPU is selected, this may require a reboot.


Before we install OpenCV for Python, we must set up the Python environment it will be compiled for.

Set up a new Python 3.7 environment using Anaconda:

'''
conda create -n GapWatch python=3.7 --file common/environments/dev/requirements.txt
'''

Make sure to activate this new environment!

'''
conda activate GapWatch
'''

Some libraries require conda forge to install:

```
conda install -c conda-forge librosa
```

Go to somewhere outside the GapWatch project directory (for example ~/Downloads)

```
git clone https://github.com/opencv/opencv.git
cd opencv/
git checkout 4.1.0
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D INSTALL_PYTHON_EXAMPLES=ON \
        -D INSTALL_C_EXAMPLES=OFF \
        -D PYTHON_EXECUTABLE=$(which python3) \
        -D BUILD_opencv_python2=OFF \
        -D CMAKE_INSTALL_PREFIX=$(python3 -c 'import sys; print(sys.prefix)') \
        -D PYTHON3_EXECUTABLE=$(which python3) \
        -D PYTHON3_INCLUDE_DIR=$(python3 -c 'from distutils.sysconfig import get_python_inc; print(get_python_inc())') \
        -D PYTHON3_PACKAGES_PATH=$(python3 -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())') \
        -D WITH_OPENCL=ON \
        -D BUILD_EXAMPLES=ON ..
```

Install this compiled OpenCV, replace 4 with the number of threads you want to run.

```
make -j8
```

Add to the anaconda environment:

'''
sudo make install
'''

Finally, add a link to the GapWatch Anaconda environment. Note that this will depend on where you installed Anaconda! See this great tutorial for more: https://jayrobwilliams.com/files/html/OpenCV_Install.html

```
cd /home/mdunlop/anaconda3/envs/GapWatch/lib/python3.7
ln -s /usr/local/python/cv2 cv2
```

If you open a python interpreter you should be able to:

1. Import cv2
2. Verify that OpenCL can be used

![Alt text](./common/assets/opencv_opencl_success.png)








## Training Image (training)

Now that the project has matured sufficiently to begin exploring more powerful models, the development environment should be able to operate on cloud hardware to enable faster training.

Ideally the latest version of tensorflow 2+ should be used for the new syntax and also better performance on new GPUs. Linode has been identified as the cloud provider of choice, offering $100 for 60 days for free which should be sufficient.

To set up the environment with Linode, one option is to use their "Stack Scripts" which are essentially shell scripts which run each time the virtual machine is setup. This is my preferred method to set up the environment as I do not have a nVidia GPU available to build docker images locally, which would inevitably result in some trial and error in the cloud, wasting time and money with every failed docker deploy.

## Intel Celeron N3060 Base Image (n3060)

It turns out that Docker is not designed with sharing WebCams or Microphones with containers in mind. Instead it is better to simply work with a single virtual environment for the machine, where I have successfully tested the camera with OpenCV.

Celeron N3060 does **NOT** support AVX instructions, so tensorflow >1.5 won't run (can compile from source but this can take up to 25 hours and is not guaranteed to work).

There are two cameras and only two threads so this will effectively be a single threaded pipeline for each camera.

Considering there are only two threads, the potential overhead of transferring image data from opencv to tensorflow and back to numpy would probably mitigate any performance gain from tensorflow in single threaded workloads.

