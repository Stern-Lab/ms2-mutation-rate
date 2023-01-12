# Mutation rate, selection, and epistasis inferred from RNA virus haplotypes via neural posterior estimation

Repository that accompanies the paper:
> Itamar Caspi, Moran Meir, Nadav Ben Nun, Uri Yakhini, Adi Stern, Yoav Ram. (2023) Mutation rate, selection, and epistasis inferred from RNA virus haplotypes via neural posterior estimation. bioRxiv. doi: [10.1101/2023.01.09.523230](https://doi.org/10.1101/2023.01.09.523230)

## Data
- Original LoopSeq fastq files of the empirical data are hosted on [SRA](https://www.ncbi.nlm.nih.gov/sra/PRJNA902661). 
- Data for analysis in this repo is hosted on [Zenodo](https://zenodo.org/record/7486851).

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

## Repo overview

This repo contains 3 notebooks for creating all the figures:
 - `empirical_data_analysis.ipynb` - Using the different inference methods to estimate posterior distributions from empirical data and create posterior predictive checks.
 - `data_analysis/data_analysis.ipynb` - Exploring the empirical data.
 - `synthetic_data_tests/synthetic_data_tests.ipynb` - Analyzing inference methods on synthetic data.
 - 

To understand the method and the pipeline, read the `inference_pipeline.py` file which outlines the entire process from creating simulations to training and testing the density estimators using functions located in the `model` directory.
 
## License

CC-BY-SA 4.0
