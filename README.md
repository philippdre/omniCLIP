# omniCLIP
omniCLIP is a Bayesian peak caller that can be applied to data from CLIP-Seq data to detect regulatory elements in RNAs. 

## Overview

[Introduction](#introduction)

[Dependencies](#dependencies)

[Installation](#installation)

[Usage](#usage)

[Examples](#examples)

[Contributors](#contributors)

[License](#license)


## Introduction
omniCLIP can call peaks for CLIP-Seq data data while accounting for confounding factors such as the gene expression and it automatically learns relevant diagnostic events from the data. Furtermore, it can leverage replicate information and model technical and biological variance.

## Dependencies
omniCLIP requires Python (v.2.7) and the following python libraries.:

* brewer2mpl (> v.1.4)
* gffutils (> v.0.8.7.1)
* h5py (> v.2.6.0)
* intervaltree (> v.2.1.0)
* matplotlib (> v.1.5.3)
* numpy (> v.1.11.3)
* prettyplotlib (> v.0.1.7)
* pysam (> v.0.9.1.4)
* scipy (> v.0.19.0)
* statsmodels (> v.0.6.1)
* scikit-learn (> v.0.18.1)
* biopython (> v.1.68)
* cython (> v.0.24.1)

## Installation

### Manual installation
The latest stable release int the ***master*** branch can be downloaded by executing:
```
$ git clone -b master https://github.com/philippdre/omniCLIP.git
```
After this the follwing comand has to be executed:
```
$ cd omniCLIP/stat
$ ./CompileCython.sh
```
Note that if your python libraries is not in the directory "/usr/include/python2.7", then you need to change in CompileCython.sh in the line 
```
gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/usr/include/python2.7 -o viterbi.so viterbi.c
``` 
-I/usr/include/python2.7" to the path to your python installation.

### Conda

in progress...

### Galaxy

in progress...

## Usage
omniCLIP requires the gene annotation to be in an SQL database. This database can be generated from a gff3 file by typing:
```
$ python data_parsing/CreateGeneAnnotDB.py INPUT.gff OUTPUT.gff.db
```

omniCLIP has the following commanline arguments
```
Required
    --annot	File where gene annotation is stored
    --genome-dir	Directory where fasta files are stored
    --clip-files	Bam-files for CLIP-libraries
    --bg-files	Bam-files for bg-libraries or files with counts per gene
    --out-dir	Output directory for results
Optional
    --restart-from-iter	restart from existing run
    --overwrite-CLIP-data	Overwrite the existing CLIP data
    --collapsed-CLIP	CLIP-reads are collapsed
    --overwrite-bg-data	Overwrite the existing CLIP data
    --collapsed-bg	bg-reads are collapsed
    --bck-var	Parse variants for background reads
    --verbosity	Verbosity
    --max-it	Maximal number of iterations
    --max-it-glm	Maximal number of iterations in GLM
    --gene-sample	Nr of genes to sample
    --no-subsample	Disabaple subsampling for parameter estimations (Warning: Leads to slow estimation)
    --filter-snps	Do not fit diagnostic events at SNP-positions
    --snp-ratio	Ratio of reads showing the SNP
    --snp-abs-cov	Absolute number of reads covering the SNP position (default = 10)
    --nr_mix_comp	Number of diagnostic events mixture components (default = 1)
    --nb-cores	Number of cores o use'
    --mask-miRNA	Mask miRNA positions
    --mask-ovrlp	Ignore overlping gene regions for diagnostic event model fitting
    --norm_class	Normalize class weights during glm fit
    --max-mismatch	Maximal number of mismatches that is allowed per read (default: 2)
    --mask_flank_mm	Do not consider mismatches in the N bp at the ends of reads for diagnostic event modelling 
```

## Examples
An example dataset can be  downloaded [here](https://ohlerlab.mdc-berlin.de/files/omniCLIP/example_data.tar.gz).
## Contributors



## License
GNU GPL lincene (v3)
