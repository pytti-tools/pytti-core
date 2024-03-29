# PyTTI-Tools: Core
[![Jupyter Book Badge](https://jupyterbook.org/badge.svg)](https://pytti-tools.github.io/pytti-book/intro.html)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pytti-tools/pytti-notebook/blob/main/pyttitools-PYTTI.ipynb)
[![DOI](https://zenodo.org/badge/452409075.svg)](https://zenodo.org/badge/latestdoi/452409075)


# Requirements

* Python 3.x
* [Pytorch](https://pytorch.org/get-started/locally/)
* CUDA-capable GPU
* OpenCV
* ffmpeg
* Python Image Library (PIL/pillow)
* git - simplifies downloading code and keeping it up to date
* gdown - simplifies downloading pretrained models
* jupyter - (Optional) Notebook interface

# Documentation

Detailed setup and usage instructions can be found here: https://pytti-tools.github.io/pytti-book/intro.html

# Setup

(2023) this works for the moment

```
git clone --recurse-submodules -j8 https://github.com/pytti-tools/pytti-core
pip install -r pytti-core/requirements.txt
pip install ./pytti-core/vendor/AdaBins
pip install ./pytti-core/vendor/CLIP
pip install ./pytti-core/vendor/GMA
pip install ./pytti-core/vendor/taming-transformers
pip install ./pytti-core
```

you're probably gonna get an error about a missing adabins checkpoint now. Download that from here and put it in the file path corresponding to the error (should be an `adabins` subfolder of your user cache that you'll probably need to mkdir):

```
wget https://github.com/hithereai/deforum-for-automatic1111-webui/releases/download/AdaBins/AdaBins_nyu.pt
```

# Development

* Rebuild, reinstall, and run tests:
    ```
    pip uninstall -y pyttitools-core; rm -rf build; pip install .
    python -m pytest --ignore=vendor -v -s
    ```
