#/bin/bash

set -e

cython viterbi.pyx -3 

gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/fast_data/drewe/software/envs/omni_devel3/include/python3.7m -o viterbi.so viterbi.c
