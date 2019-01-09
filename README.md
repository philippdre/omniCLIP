# omniCLIP
omniCLIP is a Bayesian peak caller that can be applied to data from CLIP-Seq experiments to detect regulatory elements in RNAs. 

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

## Dependencies and Requirements
omniCLIP requires Python (v.2.7) and the following python libraries:

* biopython (> v.1.68)
* brewer2mpl (> v.1.4)
* cython (> v.0.24.1)
* gffutils (> v.0.8.7.1)
* h5py (> v.2.6.0)
* intervaltree (> v.2.1.0)
* matplotlib (> v.1.5.3)
* numpy (> v.1.11.3)
* pandas (> v0.19.0)
* prettyplotlib (> v.0.1.7)
* pysam (> v.0.9.1.4)
* scikit-learn (> v.0.18.1)
* scipy (> v.0.19.0)
* statsmodels (> v.0.6.1)

Currently, omniCLIP requires a standard workstation with 32 Gb of RAM.


## Installation

### Manual installation
The latest stable release in the ***master*** branch can be downloaded by executing:
```
$ git clone -b master https://github.com/philippdre/omniCLIP.git
```
After this the follwing comand has to be executed:
```
$ cd omniCLIP/stat
$ ./CompileCython.sh
```
This compiles the cyton code for the viterbi algorithm. Note that if your python libraries is not in the directory "/usr/include/python2.7", then you need to change in CompileCython.sh in the line 
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
omniCLIP can be run as follows:

```
$ python omniCLIP.py [Commands]
```
omniCLIP has the following ***required*** commanline arguments

Argument  | Description
------------- | -------------
--annot | File where gene annotation is stored
--genome-dir | Directory where fasta files are stored
--clip-files | Bam-file for CLIP-library. The alignments need to have the NM and MD tag. 
--bg-files | Bam-file for bg-library. The alignments need to have the NM and MD tag.
--out-dir | Output directory for results


and the following ***optional*** arguments

Argument  | Description
------------- | -------------
--restart-from-iter | restart from existing run
--use-precomp-CLIP-data | Use existing fg_reads.dat file. This skips parsing the CLIP reads.
--collapsed-CLIP | CLIP-reads are collapsed
--overwrite-bg-data | Use existing bg_reads.dat data. This skips parsing the CLIP reads.
--collapsed-bg | bg-reads are collapsed
--bck-var | Parse variants for background reads
--verbosity | Verbosity
--max-it | Maximal number of iterations
--max-it-glm | Maximal number of iterations in GLM
--tmp-dir | Output directory for temporary results
--gene-sample | Nr of genes to sample
--no-subsample | Disabaple subsampling for parameter estimations (Warning: Leads to slow estimation)
--filter-snps | Do not fit diagnostic events at SNP-positions
--snp-ratio | Ratio of reads showing the SNP
--snp-abs-cov | Absolute number of reads covering the SNP position (default = 10)
--nr_mix_comp | Number of diagnostic events mixture components (default = 1)
--nb-cores | Number of cores o use'
--mask-miRNA | Mask miRNA positions
--mask-ovrlp | Ignore overlping gene regions for diagnostic event model fitting
--norm_class | Normalize class weights during glm fit
--max-mismatch | Maximal number of mismatches that is allowed per read (default: 2)
--mask_flank_mm | Do not consider mismatches in the N bp at the ends of reads for diagnostic event modelling 
--rev_strand | Only consider reads on the forward (0) or reverse strand (1) relative to the gene orientation
--use_precomp_diagmod | Use a precomputed diagnostic event model (Path to IterSaveFile.dat) 
--seed | Set a seed for the random number generators
--pv | Bonferroni corrected p-value cutoffs for peaks in bed-file


## Examples
An example dataset can be  downloaded [here](https://ohlerlab.mdc-berlin.de/files/omniCLIP/example_data.tar.gz). Extract it into the omniCLIP folder for the example below.

Then you can run omniCLIP on the example data by:
```
$ python omniCLIP.py --annot example_data/gencode.v19.annotation.chr1.gtf.db --genome-dir example_data/hg37/ --clip-files example_data/PUM2_rep1_chr1.bam --clip-files example_data/PUM2_rep2_chr1.bam --bg-files example_data/RZ_rep1_chr1.bam --bg-files example_data/RZ_rep2_chr1.bam --out-dir example_data --collapsed-CLIP --bck-var
```
This command creats the files below:

File Name | Description
------------- | -------------
pred.bed | This file contains the peaks that are signifikant after Bonferroni correction
pred.txt | This file contains all peaks 
fg_reads.dat | This file contains the parsed reads from the CLIP libraries
bg_reads.dat | This file contains the parsed reads from the background libraries
IterSaveFile.dat | This file contains the learnt parameters of the model
IterSaveFileHist.dat | This file contains the learnt parameters of the model in each iteration


## Contributors



## License
GNU GPL license (v3)
