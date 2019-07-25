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
Figures and tables are by default generated from pre-calculated benchmarking data saved in paper/benchmark_data/*.npy files

If you would like to reproduce benchmarking results from scratch, simply delete these *.npy files.

Warning: it is highly recommended that benchmarking be run on GPU(s). On CPUs they're expected to run for hours.
