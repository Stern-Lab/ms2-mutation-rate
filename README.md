# Inferring the mutation rate of an RNA bacteriophage from time-series haplotype data via neural density estimation

This is the github repository that accompanies the thesis submitted towards M.Sc. in Mathematical and Theoretical Biology
At Tel-Aviv University at Nov 27th 2022 by Itamar Caspi Under the supervision of Adi Stern and in collaboration with Yoav Ram.

The original LoopSeq fastq files of the empirical data are hosted on [SRA](https://www.ncbi.nlm.nih.gov/sra/PRJNA902661). 

Some of the data necessary to fully run the analysis in this repo is hosted on [Zenodo](https://zenodo.org/record/7486851).

This repo was only tested on Linux. Since some operations use BLAST, Windows is not supported.


## Installation

To install a Python environment, [install Anaconda](https://www.anaconda.com/products/distribution) and create a new environment:
```
conda env create -n ENV_NAME --file environment.yml 
activate ENV_NAME
```
then use pip to install [sbi](https://www.mackelab.org/sbi): 
```
python -m pip install sbi==0.17.0
```
Note that some of the notebooks require additional files which can be downloaded from [Zenodo](https://zenodo.org/record/7486851) using the direct links in the notebook or in the README.md files in their respective directories.

## Overview
#### Results
This repo contains 3 notebooks which are the goto for understanding the results of the paper and create all the graphs and more (some graphs went under minor post hoc editing):
 - `empirical_data_analysis.ipynb` - Using the different density estimators to estimate empirical data and create posterior predictive checks.
 - `data_analysis/data_analysis.ipynb` - Running and understanding the empirical data regardless of all the fancy modelling.
 - `synthetic_data_tests/synthetic_data_tests.ipynb` - Analyzing the inference results of the density estimators on synthetic data.
####  Method
To get a deeper understanding of the method and the entire simulations pipeline, the best approach would be to read the `inference_pipeline.py` file which outlines the entire process from creating simulations to training and testing the density estimators using functions located in the `model` directory.
 
