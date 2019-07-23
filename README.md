## deepCR: Cosmic Ray Rejection with Deep Learning

This repository accompanies the paper: deepCR: Cosmic Ray Rejection with Deep Learning (Zhang & Bloom (2019)), and includes code to reproduce results (figures and tables) of the paper.

deepCR is implemented separately in: https://github.com/profjsb/deepCR.

A version of deepCR is provided as a git submodule in this repo.

### to reproduce figures and tables in the paper

```
sh paper/run_all.sh
```
If you have installed deepCR, please do "pip uninstall deepCR" before executing the benchmarking code, to make sure that the submodule version is used.
Figures and tables are by default generated from pre-calculated benchmarking data saved in paper/*.npy files
If you would like to reproduce benchmarking results from scratch, simply delete these *.npy files.
Warning: it is highly recommended that benchmarking be run on GPU(s). On CPUs they're expected to run for hours.

### to train models

Training code will be available soon.

### to adapt deepCR for your own use

Please refer to https://github.com/profjsb/deepCR

example usage:
```
pip install deepCR
```
```python
from deepCR import deepCR
mdl = deepCR(mask="ACS-WFC-F606W-2-32",
	     inpaint="ACS-WFC-F606W-2-32",
             device="GPU")
mask, cleaned_image = mdl.clean(image, threshold = 0.5)
```