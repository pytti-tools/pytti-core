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
      git clone --recurse-submodules https://github.com/pytti-tools/pytti-core
      ```

    The end result should be a separate folder for each of pytti and the downloaded models. You should now have a folder structure that looks something like this:
         
            ├── pytti-notebook
            │   ├── config
            │   ├── images_out
            │   ├── pretrained
            │   ├── pytti-core
            │   └── videos

  3. Download the file `AdaBins_nyu.pt` from google drive. 

      1. Run the following command to determine the directory where you should put the model

          ```
          python -c "import os; print(os.path.expanduser('~/.cache/adabins/'))"
          ```
          on colab, this would evaluate to: `/root/.cache/adabins/`

      2. Download the model to the directory indicated by the output of the previous step. Update the command below to use the correct path.
          ```
          # If you would prefer to do this step in the browser, you can download by visiting the URL. You will have to move the file to the correct location afterwards.

          gdown -O /root/.cache/adabins/ https://drive.google.com/uc?id=1lvyZZbC9NLcS8a__YPcUP7rDiIpbRpoF

          # If you get an error saying this file is not available because it's been accessed too many times or whatever, try this alternative URL:

          gdown -O /root/.cache/adabins/ https://drive.google.com/uc?id=1zgGJrkFkJbRouqMaWArXE4WF_rhj-pxW
          ```

4. Install the cloned code into your python environment

    ```
    # Install research code
    pip install ./pytti-core/vendor/AdaBins
    pip install ./pytti-core/vendor/CLIP
    pip install ./pytti-core/vendor/GMA
    pip install ./pytti-core/vendor/taming-transformers
    pip install ./pytti-core
    ```


# Usage

For a convenient UI, open [this notebook](https://colab.research.google.com/github/pytti-tools/pytti-notebook/blob/main/pyttitools-PYTTI.ipynb) in google colab.

If you would like a notebook experience but are not using colab, please use the ["_local"](https://github.com/pytti-tools/pytti-notebook/blob/main/pyttitools-PYTTI_local.ipynb) notebook instead. The following usage notes are written with the _local notebook and command-line (CLI) use in mind.

## YAML Config Crash-Course

PYTTI uses [OmegaConf](https://omegaconf.readthedocs.io/)/[Hydra](https://hydra.cc/docs/) for configuring experiments (i.e. "runs", "renders", "generating images", etc.). In this framework, experiments are specified using text files that contain the parameters we want to use in our experiment. 

A starting set of [configuration files](https://github.com/pytti-tools/pytti-notebook/tree/main/config) is provided with the notebook repository. If you followed the setup instructions above, this `config/` folder should be in the same directory as your notebooks. If you are using the CLI, the config folder should be located in your current working directory.

### `config/default.yaml`

This file contains the default settings for all available parameters. The colab notebook can be used as a reference for how to use individual settings and what options can be used for settings that expect specific values or formats. 

Entries in this file are in the form `key: value`. Feel free to modify this file to specify defaults that are useful for you, but we recommend holding off on tampering with `default.yaml` until after you are comfortable specifying your experiments with an override config (discussed below).

### `config/conf/*.yaml`

PYTTI requires that you specify a "config node" with the `conf` argument. The simplest use here is to add a yaml file in `config/conf/` with a name that somehow describes your experiment. A `demo.yaml` is provided. 

**IMPORTANT**: The first line of any non-default YAML file you create needs to be: 

    # @package _global_

for it to work properly in the current config scheme. See the `demo.yaml` as an example [here](https://github.com/pytti-tools/pytti-notebook/blob/main/config/conf/demo.yaml#L1)

As with `default.yaml`, each parameter should appear on its own line in the form `key: value`. Starting a line with '#' is interpreted as a comment: you can use this to annotate your config file with your own personal notes, or deactivate settings you want ignored.

## Notebook Usage

The first code cell in the notebook tells PYTTI where to find your experiment configuration. The name of your configuration gets stored in the `CONFIG_OVERRIDES` variable. When you clone the notebook repo, the variable is set to `demo.yaml`. 

Executing the "RUN IT!" cell in the notebook will load the settings in `default.yaml` first, then the contents of the filename you gave to `CONFIG_OVERRIDES` are loaded and these settings will override the defaults. Therefore, you only need to explicitly specify settings you want to be different from the defaults given in `default.yaml`.

### "Multirun" in the Notebook (Intermediate)

#### Specifying multiple override configs

The `CONFIG_OVERRIDES` variable can accept a list of filenames. All files should be located in `config/conf` and follow the override configuration conventions described above. If multiple config filenames are provided, they will be iterated over sequentially.

As a simple example, let's say we wanted try three different prompts against the default settings. To achieve this, we will treat each set of prompts as its own "experiment" we want to run, so we'll need to create two override config files, one for each text prompt ("scene") we want to specify:

* `config/conf/experiment1.yaml`

      # @package _global_
      scenes: fear is the mind killer

* `config/conf/experiment2.yaml`

      # @package _global_
      scenes: it is by will alone I set my mind in motion

Now to run both of these experiments, in the second cell of the notebook we change:

    CONFIG_OVERRIDES="demo.yaml"

to

    CONFIG_OVERRIDES= [ "experiment1.yaml" , "experiment2.yaml" ]

(whitespace exaggerated for clarity.)


### Config Groups (advanced)

More details on this topic in the [hydra docs](https://hydra.cc/docs/tutorials/basic/your_first_app/config_groups/) and great examples in the [vissl docs](https://vissl.readthedocs.io/en/latest/hydra_config.html).

Hydra supports creating nested hierarchies of config files called "config groups". The hierarchy is organized using subfolders. To select a particular config file from a group, you use the same `key: value` syntax as the normal pytti parameters, except here the `key` is the name of a subdirectory you created and `value` is the name of a yaml file (without the .yaml extension) or folder in that subdirectory.

To demonstrate how this works, let's create a `motion` parameter group for storing sets of animation transformations we like to use.

First, we create a `motion` folder in `config/conf`, and add yaml files with the settings we want in that folder. So maybe something like:

* `config/conf/motion/zoom_in_slow.yaml`

      # @package _global_
      animation_mode: 3D
      translate_z_3D: 10

* `config/conf/motion/zoom_in_fast.yaml`

      # @package _global_
      animation_mode: 3D
      translate_z_3D: 100

* `config/conf/motion/zoom_out_spinning.yaml`

      # @package _global_
      animation_mode: 3D
      translate_z_3D: -50
      rotate_2D: 10

The config layout might look something like this now:

    ├── pytti-notebook/
    │   ├── config/
    |   │   ├── default.yaml
    |   │   ├── conf/
    |   │   |   ├── demo.yaml
    |   │   |   ├── experiment1.yaml
    |   │   |   ├── experiment2.yaml
    |   │   |   ├── motion/
    |   │   |   |   ├── zoom_in_slow.yaml
    |   │   |   |   ├── zoom_in_fast.yaml
    |   │   |   |   └── zoom_out_spinng.yaml

Now if we want to add one of these effects to an experiment, all we have to do is name it in the configuration like so:

* `config/conf/experiment1.yaml`

      # @package _global_
      scenes: fear is the mind killer
      motion: zoom_in_slow

## CLI usage

To e.g. run the configuration specified by `config/conf/demo.yaml`, our command would look like this:

    python -m pytti.workhorse conf=demo

Not that on the command line the convention is now `key=value` whereas it was `key: value` in the yaml files. Same keys and values work here, just need that `=` sign.

We can actually override arguments from the command line directly:

```
# to make this easier to read, I'm 
# using the line continuation character: "\"

python -m pytti.workhorse \
    conf=demo \
    steps_per_scene=300 \
    translate_x=5 \
    seed=123
```

### CLI Superpowers

A superpower commandline hydra gives us is the ability to specify multiple values for the same key, we just need to add the argument `--multirun`. For example, we can do this:

    python -m pytti.workhorse \
        --multirun \
        conf=experiment1,experiment2

This will first run `conf/experiment1.yaml` then `conf/experiment2.yaml`. Simple as that.

The real magic here is that we can provide multiple values like this *to multiple keys*, creating permutations of settings. 

Lets say that we wanted to compare our two experiments across several different random seeds:

```
python -m pytti.workhorse \
  --multirun \
  conf=experiment1,experiment2 \
  seed=123,42,1001
```

Simple as that, pytti will now run each experiment for all three seeds provided, giving us six experiments. 

This works for parameter groups as well (you may have already figured out that `conf` *is* a parameter group, so we've actually already been using this feature with parameter groups):

```
# to make this easier to read, I'm 
# using the line continuation character: "\"

python -m pytti.workhorse \
  conf=experiment1,experiment2 \
  seed=123,42,1001 \
  motion=zoom_in_slow,zoom_in_fast,zoom_and_spin
```

And just like that, we're permuting two prompts against 3 different motion transformations, and 3 random seeds. That tiny chunk of code is now generating 18 experiments for us.
