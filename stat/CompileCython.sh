#/bin/bash

set -e

cython viterbi.pyx

gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/home/pdrewe/software/anaconda2/include/python2.7 -o viterbi.so viterbi.c
