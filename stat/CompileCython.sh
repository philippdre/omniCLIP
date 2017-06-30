#/bin/bash

set -e

cython viterbi.pyx

gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/usr/include/python2.7 -o viterbi.so viterbi.c
