## deepCR: Cosmic Ray Rejection with Deep Learning

This repository accompanies the paper: deepCR: Cosmic Ray Rejection with Deep Learning (Zhang & Bloom (2019)), and includes code to reproduce results (figures and tables) of the paper.

deepCR is implemented separately in: https://github.com/profjsb/deepCR.

### to reproduce figures and tables in the paper

Tested to work on Python 3.6 and 3.7
Automatically runs on GPU if torch.cuda.is_available()
```
pip install -r requirements_pip.txt
cd paper/data/
sh generate_data.sh
cd ../
sh run_all.sh
```
Figures and tables are by default generated from pre-calculated benchmarking data saved in paper/*.npy files
If you would like to reproduce benchmarking results from scratch, simply delete these *.npy files.
Warning: it is highly recommended that benchmarking be run on GPU(s). On CPUs they're expected to run for hours.

### to adapt deepCR for your own use

Please refer to https://github.com/profjsb/deepCR

Example usage:

```python
from deepCR import deepCR
mdl = deepCR(mask="ACS-WFC-F606W-2-32",
	     inpaint="ACS-WFC-F606W-2-32",
             device="GPU")
mask, cleaned_image = mdl.clean(image, threshold = 0.5)
```