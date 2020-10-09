'''
    omniCLIP is a CLIP-Seq peak caller

    Copyright (C) 2017 Philipp Boss

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''


import sys
sys.path.append('../data_parsing/')
sys.path.append('../data_parsing/')
from scipy.special import logsumexp
from scipy.sparse import *
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.linear_model import SGDClassifier
import numpy as np
import pdb
import random
import resource
import time
import tools


#@profile 
def PredictTransistions(Counts, TransitionParameters, NrOfStates, Type = 'multi', verbosity=1):
    '''
    This function predicts the transition probabilities for a gene given the transition parameters
    '''

    TransistionProb = PredictTransistionsSimple(Counts, TransitionParameters, NrOfStates)
    
    return TransistionProb


##@profile
#@profile 
def PredictTransistionsSimple(Counts, TransitionParameters, NrOfStates, verbosity=1):
    '''
    This function predicts the transition probabilities for a gene given the transition parameters
    '''

    TransitionParametersLogReg = TransitionParameters[1]
    TransistionProb = np.ones((NrOfStates, NrOfStates, Counts.shape[1])) * np.log((1 / np.float64(NrOfStates)))

    #Genererate the features
    CovMat = GenerateFeatures(np.array(list(range(Counts.shape[1] - 1))), Counts)

    ix_nonzero = np.sum(CovMat, axis=0) > 0
    #Ceate the probailities 
    TempProb = TransitionParametersLogReg.predict_log_proba(CovMat.T).T

    NormFactor = np.ones((TempProb.shape[1]))
    for CurrentState in range(NrOfStates):
        for NextState in range(NrOfStates):
            if CurrentState == NextState:
                TransistionProb[NextState, CurrentState, 1:] = TempProb[1, :]
            else:
                TransistionProb[NextState, CurrentState, 1:] = TempProb[0, :]

        #Normalize the transition probabilities
        
        if CurrentState == 0:
            #only compute the normalizeing factore once as compuation is expensive
            
            if np.sum(ix_nonzero) > 0:
                #nonzero entries
                NormFactor[ix_nonzero] = logsumexp(TransistionProb[:, CurrentState, 1:][:,ix_nonzero], axis = 0)
            if np.sum(ix_nonzero == 0) > 0:
                #zero entries
                first_zero_pos = np.where(ix_nonzero == 0)[0][0]
                NormFactor[ix_nonzero == 0] = logsumexp(TransistionProb[:, CurrentState, 1:][:, first_zero_pos], axis = 0)

        for NextState in range(NrOfStates):
            TransistionProb[NextState, CurrentState, 1:] -= NormFactor
    
    del TempProb
    return TransistionProb


#@profile 
def FitTransistionParameters(Sequences, Background, TransitionParameters, CurrPath, C, verbosity=1):
    '''
    This function determines the optimal parameters of the logistic regression for predicting the TransitionParameters
    '''

    print('Fitting transition parameters')
    if verbosity > 0:
        print('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    
    NewTransitionParametersLogReg = FitTransistionParametersSimple(Sequences, Background, TransitionParameters, CurrPath, C, verbosity=verbosity)
    if verbosity > 0:
        print('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    
    return NewTransitionParametersLogReg



#@profile 
def FitTransistionParametersSimple(Sequences, Background, TransitionParameters, CurrPath, C, verbosity=1):
    '''
    This function determines the optimal parameters of the logistic regression for predicting the TransitionParameters
    '''

    #Generate features from the CurrPaths and the Information in the coverage
    TransitionMatrix = TransitionParameters[0]
    NewTransitionParametersLogReg = {}
    t = time.time()
    #Iterate over the possible transitions
    assert (TransitionMatrix.shape[0] > 1), 'Only two states are currently allowed'

    genes = list(CurrPath.keys())
    genes = random.sample(genes, min(len(genes), 1000))

    NrOfStates = TransitionMatrix.shape[0]
    Xs = []
    Ys = []
    SampleSame = []
    SampleOther = []
    print("Learning transition model")
    print("Iterating over genes")
    if verbosity > 0:
        print('Fitting transition parameters: I')
        print('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    for i, gene in enumerate(genes):
        if i % 1000 == 0:
            sys.stdout.write('.')
            sys.stdout.flush()
        #Get data
        Sequences_per_gene = tools.PreloadSequencesForGene(Sequences, gene)
        CovMat = tools.StackData(Sequences_per_gene, add = 'all')
        CovMat[CovMat < 0] = 0
        nr_of_samples = CovMat.shape[0]
        for CurrState in range(NrOfStates):
            for NextState in range(NrOfStates):
                #Positions where the path is in the current state
                Ix1 = CurrPath[gene][:-1] == CurrState 
                #Positions where the subsequent position path is in the "next" state
                Ix2 = CurrPath[gene][1:] == NextState
                #Positions where the path changes from the current state to the other state
                Ix = np.where(Ix1 * Ix2)[0]
                
                if np.sum(np.sum(np.isnan(CovMat)))> 0:
                    pdb.set_trace()
                CovMatIx = GenerateFeatures(Ix, CovMat)
                if np.sum(np.sum(np.isnan(CovMatIx)))> 0 or np.sum(np.sum(np.isinf(CovMatIx)))> 0:
                    pdb.set_trace()

                if CurrState == NextState:
                    if CovMatIx.shape[1] == 0:
                        CovMatIx = np.zeros((nr_of_samples, 1))
                        SampleSame.append(CovMatIx)
                    else:
                        SampleSame.append(CovMatIx)
                else:                
                    if CovMatIx.shape[1] == 0:
                        CovMatIx = np.zeros((nr_of_samples, 1))
                        SampleOther.append(CovMatIx)
                    else:
                        SampleOther.append(CovMatIx)
        del Sequences_per_gene, CovMat
        
    if verbosity > 0:
        print('Fitting transition parameters: II')
        print('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

    len_same = np.sum([Mat.shape[1] for Mat in SampleSame])
    len_other = np.sum([Mat.shape[1] for Mat in SampleOther])

    X = np.concatenate(SampleSame + SampleOther, axis =1).T
    del SampleSame, SampleOther

    #Create Y
    Y = np.hstack((np.ones((1, len_same), dtype=np.int), np.zeros((1, len_other), dtype=np.int)))[0,:].T
    classes = np.unique(Y)
    if verbosity > 0:
        print('Fitting transition parameters: III')
        print('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    n_iter = max(5, np.ceil(10**6 / Y.shape[0]))
    

    NewTransitionParametersLogReg = SGDClassifier(loss="log", max_iter = n_iter)
    ix_shuffle = np.arange(X.shape[0])
    for n in range(n_iter):
        np.random.shuffle(ix_shuffle)
        for batch_ix in np.array_split(ix_shuffle, 50):
            NewTransitionParametersLogReg.partial_fit(X[batch_ix,:], Y[batch_ix], classes=classes)

    if verbosity > 0:
        print('Fitting transition parameters: IV')
        print('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    del Ix1, Ix2,  Ix, X, Y, Xs, Ys 
    if verbosity > 0:
        print('Done: Elapsed time: ' + str(time.time() - t))

    return NewTransitionParametersLogReg




#@profile 
def GenerateFeatures(Ix, CovMat):
    '''
    This funnction generates, for a set of positions, the features for the logistic regression from the Coverage matrix
    '''

    FeatureMatrix = np.log(1 + CovMat[:, Ix])
    return FeatureMatrix
