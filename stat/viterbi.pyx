#!/usr/bin/env python2.7

cimport cython
from cython.view cimport array as cvarray
import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)


def viterbi(np.ndarray[DTYPE_t, ndim=2] EmmisionProbabilites, np.ndarray[DTYPE_t, ndim=3] TransistionProbabilities, np.ndarray[DTYPE_t, ndim=2] PriorProbabilities):
	"""
	This function determines the most likely path using the Viterbi algorithm
	"""
	cdef int i, j, k, CurrArgmax
	cdef int [:,:] TraceBack
	cdef unsigned int SeqLen = EmmisionProbabilites.shape[1]
	cdef unsigned int NrOfStates = EmmisionProbabilites.shape[0]
	#cdef int [:] Path
	cdef np.ndarray[np.int_t, ndim=1] Path = np.zeros(SeqLen, dtype=np.int)
	cdef double [:,:] V
	cdef double CurrMax, Temp
	#cdef double LogLik
	cdef LogLik = np.float64
	cdef unsigned int D_SIZE = sizeof(double)
	cdef unsigned int I_SIZE = sizeof(int)

	TraceBack = cvarray(shape=(NrOfStates, SeqLen + 1), itemsize=I_SIZE, format='i', mode="c")
	#Path = np.zeros(SeqLen, dtype=np.int64)
	V = cvarray(shape=(NrOfStates, SeqLen + 1), itemsize=D_SIZE, format='d', mode="c")


	#First step of Viterbi
	for j in xrange(NrOfStates):
		V[j, 0] = EmmisionProbabilites[j, 0] + PriorProbabilities[j, 0]

	#Iterate over the positions
	for i in xrange(0, SeqLen):
		for j in xrange(NrOfStates):
			CurrArgmax = 0
			CurrMax = V[0, i] + TransistionProbabilities[0, j, i]
			for k in xrange(1, NrOfStates):
				Temp = V[k, i] + TransistionProbabilities[k, j, i]
				if CurrMax < Temp:
					CurrMax = Temp
					CurrArgmax = k

			TraceBack[j, i] = CurrArgmax
			V[j, i + 1] = V[CurrArgmax, i] + EmmisionProbabilites[j, i] + TransistionProbabilities[CurrArgmax, j, i]

	#Perform traceback

	CurrArgmax = 0
	CurrMax = V[0, SeqLen]
	for k in xrange(1, NrOfStates):
		Temp = V[k, SeqLen]
		if CurrMax < Temp:
			CurrMax = Temp
			CurrArgmax = k

	k = CurrArgmax
	Path[SeqLen - 1] = k
	for i in xrange(SeqLen - 2, -1, -1):
		k = TraceBack[k, i]
		Path[i] = k

	LogLik = V[0, SeqLen]
	
	return Path, LogLik

