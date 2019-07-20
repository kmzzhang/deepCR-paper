## deepCR: Deep Learning Based Cosmic Ray Removal for Astronomical Images

This repository accompanies the paper: deepCR: Cosmic Ray Rejection with Deep Learning (Zhang & Bloom (2019)), and includes code to reproduce results of the paper.

deepCR is implemented separately in https://github.com/profjsb/deepCR. We have cloned an early version as submodule in this directory.

### to adapt deepCR for your own use

refer to https://github.com/profjsb/deepCR

```
pip install deepCR
```

### to reproduce figures and tables in the paper

```
sh run_all.sh
```
To avoid potential inconsistancies caused by updated models and/or code, please do "pip uninstall deepCR" so that the version in this repo is used.

### to re-train models

Training code will be available soon.
```
python train.py ...
```

### Contributing

We are very interested in getting bug fixes, new functionality, and new models from the community.
Please refer to  https://github.com/profjsb/deepCR