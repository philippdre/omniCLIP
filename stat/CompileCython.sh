#/bin/bash

set -e

cython viterbi.pyx -3 

gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/home/pdrewe/software/anaconda2/envs/omniCLIP_py3/include/python3.7m/ -o viterbi.so viterbi.c
