# Edge

The purpose of this directory is to abstract the complications of deploying models on different hardware as much as possible. This allows the remainder of the project to focus data processing and model development.

Since the final hardware this project will be deployed on is far from set in stone, I will attempt to prevent any avoidable constraints on the models themselves.

There are situations where this is unavoidable, such as on the Intel Celeron N3060 which is a budget CPU that does not support AVX instructions which implies that certain models cannot be run on this device. Similarly, there can be problems when installing python packages on ARM platforms which work fine on x86.

# x86/64 - HP Stream 11-y-5-na

## Specifications:

|                  | HP Stream 11-y-5-na |
|------------------|---------------------|
| CPU              | Intel Celeron N3060 |
| Cores            | 2                   |
| Threads          | 2                   |
| RAM              | 2 GB                |
| AVX Instructions | None                |

## Deployment Method:

Initially I had planned to use Docker containers to:
* Streamline deployment with to autobuild on git push
* Ease of deployment to mass numbers of machines (run a git pull and youre done)
* Not having to restart host in event of crash

However after two days of experimentation, I have confirmed that not all WebCams will support communication with the Docker containers. This is unsurprising as Docker containers were not designed with WebCam support.

Fortunately, OpenCV on bare-metal Ubuntu does support the WebCam. Setup will use Anaconda to manage virtual environements, where the same environment can be used for training and testing.

### Virtual Environment Setup

We will use Anaconda to manage virtual environments.

```
conda create --name GapWatch --file requirements.txt
conda activate GapWatch
conda install -c conda-forge librosa=0.7.2
```

Then install Open-CV through pip (because the Conda version of this library is currently broken (27 May 2020))

```
pip install opencv-python=4.2.0.34
```

