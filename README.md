# pytti
python text to image

# Requirements

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

# Setup
## Automated Setup

---
NB: AUTOMATED SETUP IS ONLY CURRENTLY SUPPORTED ON GOOGLE COLAB. Local use still requires manual setup.
---

The simplest way to perform the setup is to download the notebook and let it do the setup for you. This has the added benefit of providing you with a convenient interface for interacting with pytti. Both git and jupyter are required for the notebook setup to work.

To get the notebook:

```
git clone https://github.com/pytti-tools/pytti-notebook
```

Skip to the cell labeled "install everything else!"

## Manual Setup

0. Clone the pytti-notebook project and change directory into it.

The pytti-notebook folder will be our root directory for the rest of the setup sequence.

    # Download the code and create the pytti-notebook directory 
    git clone https://github.com/pytti-tools/pytti-notebook

    # Navigate into the directory
    cd pytti-notebook

git clone https://github.com/pytti-tools/pytti-notebook

1. Setup a local environment that emulates google colab

   In addition to providing free GPUs, Google colab is helpful because a lot of stuff comes pre-installed. The following steps configure an environment containing some of the tools that come pre-installed in colab. This setup procedure should significantly facilitate running any colab notebook locally, not just pytti.

   In the near future, this sequence will hopefully be replaced with a single step.

    1. Install python

        We strongly recommend installing python via the free anaconda distribution. If you have installed the full Anaconda distribution, you may skip to step 1.7.

        Get Anaconda here: https://www.anaconda.com/products/individual

    2. Create and acivate a new environment

        One of Conda's primary capabilities is creating "virtual environments,' which allows us to compartmentalize dependencies to ensure projetcs with different dependencies don't conflict with each other. 

        To create a new conda environment named: `pytti-tools`

            conda create -n pytti-tools

        We now need to "activate" this environment to use it, like so:

            conda activate pytti-tools

        The environment name shows up at the beginning of the line in the terminal. After running this command, it should have changed from `(base)` to `(pytti-tools)`. The installation steps that follow will now install into our new environment only. 

        When you are done working in this environment (you are not done yet, this is just informational), run

            conda deactivate

        You should see the environment name change back to `(base)`. The tools you installed into the pytti-tools environment will no longer be accessible, so you won't have to worry about breaking the environment by installing conflicting versions of dependencies when you are not using it.

        Sorry, some of this just gonna be like that.
        
        EXPERIMENTAL: An alternative to manually creating the environment is described here: https://github.com/pytti-tools/pytti-core/issues/32  
        If you are succcessful, you should be able to skip down step 2 ("Download the code for pytti and the models it depends on").  
        If you attempt this, whether you are successful or not: please report your experience in the linked github issue.

    3. Install Pytorch

        Follow the installation steps for installing pytorch with CUDA/GPU support here: https://pytorch.org/get-started/locally/

        If you're on windows and following the steps for conda, this is probably what you need:

            conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

    4. Install tensorflow

        To install with conda:

            conda install tensorflow-gpu

    5. Install OpenCV

        To install with conda:

            conda install -c conda-forge opencv
    
    6. Install the Python Image Library (aka pillow/PIL)

        To install with conda:

           conda install -c conda-forge pillow

    7. You get the idea. Here're some more conda installations you'll need.

            conda install -c conda-forge imageio
            conda install -c conda-forge pytorch-lightning
            conda install -c conda-forge kornia
            conda install -c huggingface transformers
            conda install scikit-learn pandas

    8. Install pip dependencies

            pip install jupyter gdown loguru einops seaborn PyGLM ftfy regex tqdm hydra-core adjustText exrex bunch matplotlib-label-lines


2. Download the code for pytti and the models it depends on. 
    
    We recommend using `git` for this step, but you could also download manually. To install git, follow the instructions here and use all the defaults in the installer: https://git-scm.com/book/en/v2/Getting-Started-Installing-Git 


      ```
      # Download this codebase
      git clone --recursive-submodules https://github.com/pytti-tools/pytti-core
      ```

    The end result should be a separate folder for each of pytti and the downloaded models. You should now have a folder structure that looks something like this:
         
            ├── pytti-notebook
            │   ├── config
            │   ├── images_out
            │   ├── pretrained
            │   ├── pytti-core
            │   └── videos

  3. Download the file `AdaBins_nyu.pt` from google drive. 

          # If you would prefer to do this step in the browser, just visit the URL.

          gdown -O ./pretrained/ https://drive.google.com/uc?id=1lvyZZbC9NLcS8a__YPcUP7rDiIpbRpoF

          # If you get an error saying this file is not available because it's been accessed too many times or whatever, try this alternative URL:

          gdown -O ./pretrained/ https://drive.google.com/uc?id=1zgGJrkFkJbRouqMaWArXE4WF_rhj-pxW

      If you downloaded manually, move `AdaBins_nyu.pt` into the `pytti-notebook/pretrained` subdirectory.

4. Install the cloned code into your python environment

    ```
    # Install research code
    pip install ./pytti-core/vendor/AdaBins
    pip install ./pytti-core/vendor/CLIP
    pip install ./pytti-core/vendor/GMA
    pip install ./pytti-core
    ```


# Usage

## Quick Hydra/yaml tutorial

## Notebook Usage

## CLI usage
