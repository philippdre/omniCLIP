# omniCLIP
omniCLIP is a Bayesian peak caller that can be applied to data from CLIP-Seq experiments to detect regulatory elements in RNAs.

## Overview

[Introduction](#introduction)

[Dependencies](#dependencies)

[Installation](#installation)

[Usage](#usage)

[Contributors](#contributors)

[License](#license)


## Introduction
omniCLIP can call peaks for CLIP-Seq data data while accounting for confounding factors such as the gene expression and it automatically learns relevant diagnostic events from the data. Furtermore, it can leverage replicate information and model technical and biological variance.

## Dependencies and Requirements
omniCLIP requires Python (v.3.7) and the libraries described in the `environment.yml` file. All required dependencies can be installed using *conda* by executing the following in the main project directory :
```
$ conda env create -f environment.yml
```

The environment then needs to be activated in order to run omniCLIP :
```
$ conda activate omniEnv
```

## Installation

### Manual installation
The latest stable release in the ***master*** branch can be downloaded by executing:
```
$ git clone -b master https://github.com/simojoe/omniCLIP.git
```
After this the following command has to be executed:
```
$ python3 setup.py
```
This compiles the Cython code for the viterbi algorithm.

## Usage
An omniCLIP analysis is run into four different steps :
- Generating the annotation database
- Parsing the background RNA-seq files
- Parsing the CLIP files
- Running the omniCLIP algorithm

The following is an example of the commands. The commands in this example only show the **required** arguments for the analysis. The following files are necessary to run an analysis.

File name  | Description
------------- | -------------
$GFF_file | Genome annotation file
$GENOME_dir | Directory containing FASTA files with each of the chromosomes sequence
$BG_file[1,2,...] | BAM files for the background library. The alignments need to have the MD and NM tags.
$CLIP_file[1,2,...] | BAM files for the CLIP library. The alignments need to have the MD and NM tags.

The following files will be created.

File name  | Description
------------- | -------------
$DB_file | SQL database of the genome annotation.
$BG_dat | H5PY file of the parsed background library.
$CLIP_dat | H5PY file of the parsed CLIP library.
$OUT_dir | Directory containing the final results


### 1. Generating the annotation database
```
$ python3 omniCLIP.py generateDB \
--gff-file $GFF_file --db-file $DB_file
```

### 2. Parsing the background RNA-seq files
```
$ python3 omniCLIP.py parsingBG \
--db-file $DB_file --genome-dir $GENOME_dir \
--bg-files $BG_file1 --bg-files $BG_file2 \
--out-file $BG_dat
```

### 3. Parsing the CLIP files
```
$ python3 omniCLIP.py parsingCLIP \
--db-file $DB_file --genome-dir $GENOME_dir \
--clip-files $CLIP_file1 --clip-files $CLIP_file2 \
--out-file $CLIP_dat
```


### 4. Running the omniCLIP algorithm
```
$ python3 omniCLIP.py run_omniCLIP \
--db-file $DB_file --bg-dat $BG_dat --clip-dat $CLIP_dat \
--out-dir $OUT_dir
```

### Optional arguments
For the full list of optional arguments of the different step, consult the help of the commands using :
```
$ python3 omniCLIP.py parsingBG --help
```

## Contributors



## License
GNU GPL license (v3)
