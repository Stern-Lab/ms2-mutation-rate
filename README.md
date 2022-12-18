# Inferring the mutation rate of an RNA bacteriophage from time-series haplotype data via neural density estimation

This is the github repository that accompanies the thesis submitted towards M.Sc. in Mathematical and Theoretical Biology
At Tel-Aviv University at Nov 27th 2022 by Itamar Caspi Under the supervision of Adi Stern and in collaboration with Yoav Ram.

The original LoopSeq fastq files of the empirical data are hosted on [SRA](https://www.ncbi.nlm.nih.gov/sra/PRJNA902661). 

Some of the data necessary to fully run the analysis in this repo is hosted on [Zenodo](https://zenodo.org/record/7307532).

This repo was only tested on Linux. Since some operations use BLAST, Windows is not supported.


## Installation

To install a Python environment, [install Anaconda](https://www.anaconda.com/products/distribution) and create a new environment:
```
conda env create -n ENV_NAME --file environment.yml 
activate ENV_NAME
```
then use pip to install [sbi](https://www.mackelab.org/sbi): 
```
python -m pip install sbi
```

