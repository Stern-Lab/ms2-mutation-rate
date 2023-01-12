# Mutation rate, selection, and epistasis inferred from RNA virus haplotypes via neural posterior estimation

Repository that accompanies the paper:
> Itamar Caspi, Moran Meir, Nadav Ben Nun, Uri Yakhini, Adi Stern, Yoav Ram. (2023) Mutation rate, selection, and epistasis inferred from RNA virus haplotypes via neural posterior estimation. bioRxiv. doi: [10.1101/2023.01.09.523230](https://doi.org/10.1101/2023.01.09.523230)

## Abstract
RNA viruses are particularly notorious for their high levels of genetic diversity, which is generated through the forces of mutation and natural selection. However, disentangling these two forces is a considerable challenge, and this may lead to widely divergent estimates of viral mutation rates, as well as difficulties in inferring fitness effects of mutations. Here, we develop, test, and apply an approach aimed at inferring the mutation rate and key parameters that govern natural selection, from haplotype sequences covering full length genomes of an evolving virus population. Our approach employs neural posterior estimation, a computational technique that applies simulation-based inference with neural networks to jointly infer multiple model parameters. We first tested our approach on synthetic data simulated using different mutation rates and selection parameters while accounting for sequencing errors. Reassuringly, the inferred parameter estimates were accurate and unbiased. We then applied our approach to haplotype sequencing data from a serial-passaging experiment with the MS2 bacteriophage. We estimated that the mutation rate of this phage is around 0.2 mutations per genome per replication cycle (95% highest density interval: 0.051-0.56). We validated this finding with two different approaches based on single-locus models that gave similar estimates but with much broader posterior distributions. Furthermore, we found evidence for reciprocal sign epistasis between four strongly beneficial mutations that all reside in an RNA stem-loop that controls the expression of the viral lysis protein, responsible for lysing host cells and viral egress. We surmise that there is a fine balance between over and under-expression of lysis that leads to this pattern of epistasis. To summarize, we have developed an approach for joint inference of the mutation rate and selection parameters fromfull haplotype data with sequencing errors, andused it to reveal features governing MS2 evolution. 

## Data availability
- Original LoopSeq fastq files of the empirical data are hosted on [SRA](https://www.ncbi.nlm.nih.gov/sra/PRJNA902661). 
- Data for analysis in this repo is hosted on [Zenodo](https://zenodo.org/record/7486851).

## Installation

<i>This repo was tested on Linux only. Since some operations use BLAST, Windows is not supported.</i>

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
 - [empirical_data_analysis.ipynb](https://github.com/Stern-Lab/ms2-mutation-rate/blob/main/empirical_data_inference.ipynb) - Using the different inference methods to estimate posterior distributions from empirical data and create posterior predictive checks.
 - [data_analysis/data_analysis.ipynb](https://github.com/Stern-Lab/ms2-mutation-rate/blob/main/data_analysis/data_analysis.ipynb) - Exploring the empirical data.
 - [synthetic_data_tests/synthetic_data_tests.ipynb](https://github.com/Stern-Lab/ms2-mutation-rate/blob/main/synthetic_data_tests/synthetic_data_tests.ipynb) - Analyzing inference methods on synthetic data.

To understand the method and the pipeline, read the [inference_pipeline.py](https://github.com/Stern-Lab/ms2-mutation-rate/blob/main/inference_pipeline.py) file which outlines the entire process from creating simulations to training and testing the density estimators using functions located in the [model](https://github.com/Stern-Lab/ms2-mutation-rate/tree/main/model) directory.

We compare our posterior estimation to the posterior estimation generated by [FITS](https://academic.oup.com/ve/article/5/1/vez011/5512690?login=true). FITS is a single-locus Wright-Fisher model applied to all synonymous SNVs, assuming they are all neutral, and uses rejection sampling to approximate the posterior distribution of model parameters. The [FITS directory](https://github.com/Stern-Lab/ms2-mutation-rate/tree/main/FITS) includes all files necessary to get a posterior estimation from FITS but in order to run it FITS needs to be installed.

The [density_estimators](https://github.com/Stern-Lab/ms2-mutation-rate/tree/main/density_estimators) directory contains all the neural posterior estimators we used to generate the posterior distributions.
 
## License

CC-BY-SA 4.0
