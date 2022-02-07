# pytti
python text to image

# Setup

## Requirements

* Python 3.x
* [Pytorch](https://pytorch.org/get-started/locally/)
* Cuda-capable GPU
* OpenCV
* ffmpeg
* Python Image Library (PIL/pillow)

**Optional**  
* git - simplifies downloading code and keeping it up to date
* jupyter - Notebook interface
* gdown - simplifies downloading pretrained models

## Installation

### Automated Install

The simplest way to perform the setup is to download the notebook and let it do the setup for you. This has the added benefit of providing you with a convenient interface for interacting with pytti. Both git and jupyter are required for the notebook setup to work.

To get the notebook:

```
git clone https://github.com/pytti-tools/pytti-notebook
```

Skip to the cell labeled "install everything else!"

### Manual Install

Perform all of these steps in the location on your computer where you plan to use these tools. We recommend creating a new empty folder and performing these steps inside that folder.

1. Download the code for pytti and the models it depends on. 
  * We recommend using `git` for this step, but you could also download manually.
  * The end result should be a separate folder for each of pytti and the downloaded models
    ```
    # Download this codebase
    git clone https://github.com/pytti-tools/pytti-core

    # Download research code from reference repositories
    git clone https://github.com/openai/CLIP
    git clone https://github.com/CompVis/taming-transformers

    # Download research code modified to integrate better with PYTTI
    git clone https://github.com/pytti-tools/AdaBins
    git clone https://github.com/pytti-tools/GMA

2. Add the libraries to your python environment

  * It is generally recommended you do this in an isolated python environment to ensure there are no conflicts between PYTTI dependencies and dependencies for other python tools you may use now or in the future. 
  * We recommended using [Anaconda](https://docs.anaconda.com/anaconda/) or [venv](https://docs.python.org/3/library/venv.html) for creating and managing these environments. 

    ```
    # Install python libraries ...this might cause version conflicts :(
    #Maybe skip this step for now. 
    #pip install -r pytti-core/requirements.txt

    # Install research code
    pip install ./CLIP
    pip install ./AdaBins
    pip install ./GMA

    # Install pytti
    pip install ./pytti-core
    ```

3. Create an empty folder named "pretrained" 

   ```
   mkdir pretrained
   ```

4. Download pretrained AdaBins models for depth estimation

    ```
    !gdown -O ./pretrained/ https://drive.google.com/uc?id=1lvyZZbC9NLcS8a__YPcUP7rDiIpbRpoF

    # Backup download location in case the first is throttled
    if [ -f pretrained/AdaBins_nyu.pt ]
    then
        !gdown -O ./pretrained/ https://drive.google.com/uc?id=1zgGJrkFkJbRouqMaWArXE4WF_rhj-pxW
    fi
    ```

5. (optional) Create folders in which generated images and videos will be outputted. Creating the following two folders is recommended to be consistent with the notebook setup:

    ```
    mkdir images_out
    mkdir videos
    ```
