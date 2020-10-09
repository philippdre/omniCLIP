#/bin/bash

set -e

cython viterbi.pyx -3 

gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/usr/include/python3.8 -o viterbi.so viterbi.c
