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
from scipy.misc import logsumexp
from scipy.sparse import *
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.linear_model import SGDClassifier
import numpy as np
import pdb
import random
import resource
import time
import tools


def PredictTransistions(Counts, TransitionParameters, NrOfStates, Type = 'multi'):
    '''
    This function predicts the transistion probabilities for a gene given the transition parameters
    '''

    if Type == 'binary':
        TransistionProb = PredictTransistionsSimple(Counts, TransitionParameters, NrOfStates)
    elif Type == 'binary_bck':
        TransistionProb = PredictTransistionsSimple(Counts, TransitionParameters, NrOfStates)
    elif Type == 'unif':
        TransistionProb = PredictTransistionsUnif(Counts, TransitionParameters, NrOfStates)
    elif Type == 'unif_bck':
        TransistionProb = PredictTransistionsUnif(Counts, TransitionParameters, NrOfStates)
    elif Type == 'multi':
        TransistionProb = PredictTransistionsMultinomialSeparate(Counts, TransitionParameters, NrOfStates)
    else :# Type == 'complete':
        TransistionProb = PredictTransistionsMultinomial(Counts, TransitionParameters, NrOfStates)

    return TransistionProb


def PredictTransistionsUnif2(Counts, TransitionParameters, NrOfStates):
    '''
    This function predicts the transistion probabilities for a gene given the transition parameters
    '''

    TransitionParametersLogReg = TransitionParameters[1]
    TransistionProb = np.log(np.ones((NrOfStates, NrOfStates, Counts.shape[1])) * (1 / np.float64(NrOfStates)))

    #Genererate the features
    CovMat = GenerateFeatures(np.array(range(Counts.shape[1] - 1)), Counts)
    for CurrentState in xrange(NrOfStates):
        #Ceate the probailities for the current state
        TempProb = TransitionParametersLogReg[CurrentState].predict_log_proba(CovMat.T).T
        TransistionProb[CurrentState, CurrentState, 1:] = TempProb[0, :]
        TransistionProb[1 - CurrentState, CurrentState, 1:] = TempProb[1, :]
        del TempProb

    return TransistionProb


def PredictTransistionsUnif(Counts, TransitionParameters, NrOfStates):
    '''
    This function predicts the transistion probabilities for a gene given the transition parameters
    '''

    TransitionParametersLogReg = TransitionParameters[1]
    TransistionProb = np.log(np.ones((NrOfStates, NrOfStates, Counts.shape[1])) * (1 / np.float64(NrOfStates)))

    #Genererate the features
    CovMat = GenerateFeatures(np.array(range(Counts.shape[1] - 1)), Counts)
    
    CurrClass = 0
    TempProb = TransitionParametersLogReg.predict_log_proba(CovMat.T).T
    for CurrentState in range(NrOfStates):
        for NextState in range(NrOfStates):
            if CurrentState == NextState:
                TransistionProb[CurrentState, CurrentState, 1:] = TempProb[1, :]
            else:
                TransistionProb[CurrentState, NextState, 1:] = TempProb[0, :] + np.log(0.5)
            CurrClass += 1      
            
    del TempProb
    return TransistionProb


def PredictTransistionsMultinomialSeparate(Counts, TransitionParameters, NrOfStates):
    '''
    This function predicts the transistion probabilities for a gene given the transition parameters
    '''

    TransitionParametersLogReg = TransitionParameters[s1]
    TransistionProb = np.log(np.ones((NrOfStates, NrOfStates, Counts.shape[1])) * (1 / np.float64(NrOfStates)))

    #Genererate the features
    CovMat = GenerateFeatures(np.array(range(Counts.shape[1] - 1)), Counts)
    
    CurrClass = 0
    for CurrentState in range(NrOfStates):
        CurrClass = 0
        #Ceate the probailities for the current state
        TempProb = TransitionParametersLogReg[CurrentState].predict_log_proba(CovMat.T).T
        for NextState in range(NrOfStates):
            TransistionProb[NextState, CurrentState, 1:] = TempProb[CurrClass, :]
            CurrClass += 1      

        #Normalize the transition probabilities 
        NormFactor = np.log(np.sum(np.exp(TransistionProb[:, CurrentState, 1:]), axis = 0))
        for NextState in range(NrOfStates):
            TransistionProb[NextState, CurrentState, 1:] -= NormFactor
    
    del TempProb
    return TransistionProb

#@profile
def PredictTransistionsSimple(Counts, TransitionParameters, NrOfStates):
    '''
    This function predicts the transistion probabilities for a gene given the transition parameters
    '''

    TransitionParametersLogReg = TransitionParameters[1]
    TransistionProb = np.log(np.ones((NrOfStates, NrOfStates, Counts.shape[1])) * (1 / np.float64(NrOfStates)))

    #Genererate the features
    CovMat = GenerateFeatures(np.array(range(Counts.shape[1] - 1)), Counts)

    #Ceate the probailities 
    TempProb = TransitionParametersLogReg.predict_log_proba(CovMat.T).T

    for CurrentState in range(NrOfStates):
        for NextState in range(NrOfStates):
            if CurrentState == NextState:
                TransistionProb[NextState, CurrentState, 1:] = TempProb[1, :]
            else:
                TransistionProb[NextState, CurrentState, 1:] = TempProb[0, :]

        #Normalize the transition probabilities
        NormFactor = logsumexp(TransistionProb[:, CurrentState, 1:], axis = 0)
        for NextState in range(NrOfStates):
            TransistionProb[NextState, CurrentState, 1:] -= NormFactor
    
    del TempProb
    return TransistionProb


def PredictTransistionsSimpleBck(Counts, TransitionParameters, NrOfStates):
    '''
    This function predicts the transistion probabilities for a gene given the transition parameters
    '''

    TransitionParametersLogReg = TransitionParameters[1]
    TransistionProb = np.log(np.ones((NrOfStates, NrOfStates, Counts.shape[1])) * (1 / np.float64(NrOfStates)))

    #Genererate the features
    CovMat = GenerateFeatures(np.array(range(Counts.shape[1] - 1)), Counts)
    #Ceate the probailities 
    TempProb = TransitionParametersLogReg.predict_log_proba(CovMat.T).T
    #pdb.set_trace()
    for CurrentState in range(NrOfStates):
        for NextState in range(NrOfStates):
            if CurrentState == NextState:
                TransistionProb[NextState, CurrentState, 1:] = TempProb[1, :]
            else:
                TransistionProb[NextState, CurrentState, 1:] = TempProb[0, :]
        #Normalize the transition probabilities
        NormFactor = logsumexp(TransistionProb[:, CurrentState, 1:], axis = 0)
        for NextState in range(NrOfStates):
            TransistionProb[NextState, CurrentState, 1:] -= NormFactor
    
    del TempProb
    return TransistionProb


def PredictTransistionsMultinomialSeparateManual(Counts, TransitionParameters, NrOfStates):
    '''
    This function predicts the transistion probabilities for a gene given the transition parameters
    '''

    TransitionParametersLogReg = TransitionParameters[1]
    TransistionProb = np.log(np.ones((NrOfStates, NrOfStates, Counts.shape[1])) * (1 / np.float64(NrOfStates)))

    #Genererate the features
    CovMat = GenerateFeatures(np.array(range(Counts.shape[1] - 1)), Counts)

    CurrClass = 0
    for CurrentState in range(NrOfStates):
        CurrClass = 0
        #Ceate the probailities for the current state

        TempProb = TransitionParametersLogReg[CurrentState][0].predict_log_proba(CovMat.T).T
        for NextState in TransitionParametersLogReg[CurrentState][1]:
            TransistionProb[NextState, CurrentState, 1:] = TempProb[CurrClass, :]
            CurrClass += 1      

        NormFactor = np.log(np.sum(np.exp(TransistionProb[:, CurrentState, 1:]), axis = 0))
        for NextState in range(NrOfStates):
            TransistionProb[NextState, CurrentState, 1:] -= NormFactor
        
    del TempProb
    return TransistionProb


def PredictTransistionsMultinomial(Counts, TransitionParameters, NrOfStates):
    '''
    This function predicts the transistion probabilities for a gene given the transition parameters
    '''

    TransitionParametersLogReg = TransitionParameters[1]
    TransistionProb = np.log(np.ones((NrOfStates, NrOfStates, Counts.shape[1])) * (1 / np.float64(NrOfStates)))

    #Genererate the features
    CovMat = GenerateFeatures(np.array(range(Counts.shape[1] - 1)), Counts)
    CovMat[CovMat < 0] = 0
    #Ceate the probailities for the current state
    TempProb = TransitionParametersLogReg.predict_log_proba(CovMat.T).T
    #pdb.set_trace()
    CurrClass = 0
    for CurrentState in range(NrOfStates):
        for NextState in range(NrOfStates):
            TransistionProb[NextState, CurrentState, 1:] = TempProb[CurrClass, :]
            CurrClass += 1      
        #Normalize the transition probabilities        
        NormFactor = np.log(np.sum(np.exp(TransistionProb[:, CurrentState, 1:]), axis = 0))
        for NextState in range(NrOfStates):
            TransistionProb[NextState, CurrentState, 1:] -= NormFactor
        
    del TempProb
    

    return TransistionProb


def FitTransistionParameters(Sequences, Background, TransitionParameters, CurrPath, C, Type = 'multi'):
    '''
    This function determines the optimal parameters of the logistic regression for predicting the TransitionParameters
    '''

    print 'Fitting transistion parameters'
    print 'Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    
    if Type == 'binary':
        NewTransitionParametersLogReg = FitTransistionParametersSimple(Sequences, Background, TransitionParameters, CurrPath, C)
    elif Type == 'binary_bck':
        NewTransitionParametersLogReg = FitTransistionParametersSimpleBck(Sequences, Background, TransitionParameters, CurrPath, C)
    elif Type == 'unif':
        NewTransitionParametersLogReg = FitTransistionParametersUnif(Sequences, Background, TransitionParameters, CurrPath, C)
    elif Type == 'unif_bck':
        NewTransitionParametersLogReg = FitTransistionParametersUnifBck(Sequences, Background, TransitionParameters, CurrPath, C)
    elif Type == 'multi':
        NewTransitionParametersLogReg = FitTransistionParametersMultinomialSeparate(Sequences, Background, TransitionParameters, CurrPath, C)
    else :# Type == 'complete':
        NewTransitionParametersLogReg = FitTransistionParametersBinary(Sequences, Background, TransitionParameters, CurrPath, C)
    print 'Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    
    return NewTransitionParametersLogReg


def FitTransistionParametersUnif2(Sequences, Background, TransitionParameters, CurrPath, C):
    '''
    This function determines the optimal parameters of the logistic regression for predicting the TransitionParameters
    '''

    #Generate features from the CurrPaths and the Information in the coverage
    TransitionMatrix = TransitionParameters[0]
    NewTransitionParametersLogReg = {}
    t = time.time()

    #Iterate over the possible transistions
    assert (TransitionMatrix.shape[0] > 1), 'Only two states are currently allowed'
    for CurrState in range(TransitionMatrix.shape[0]):
        print "Learning transistion model for State " + str(CurrState)
        SampleSame = []
        SampleOther = []

        #Iterate over the genes
        print 'Loading data'
        for i, gene in enumerate(CurrPath.keys()):
            if i % 1000 == 0:
                sys.stdout.write('.')
                sys.stdout.flush()
            #Positions where the path is in the current state
            Ix1 = CurrPath[gene][:-1] == CurrState 
            #Positions where the subsequent position path is in the other state
            Ix2 = CurrPath[gene][1:] == (1 - CurrState)
            #Positions where the path changes from the current state to the other state
            Ix = np.where(Ix1 * Ix2)[0]
            #CovMat = Sequences[gene]['CovNr'].toarray()
            Sequences_per_gene = PreloadSequencesForGene(Sequences, gene)
            CovMat = tools.StackData(Sequences_per_gene, add = 'all')
            CovMat = GenerateFeatures(Ix, CovMat)
            SampleOther.append(CovMat[:, np.sum(CovMat, axis = 0) > 0])

            #Positions where the path is in the current state
            Ix1 = CurrPath[gene][:-1] == CurrState 
            #Positions where the subsequent position path is in the same state
            Ix2 = CurrPath[gene][1:] == CurrState
            #Positions where the path stays in the current stae
            Ix = np.where(Ix1 * Ix2)[0]
            #CovMat = Sequences[gene]['CovNr'].toarray()
            Sequences_per_gene = PreloadSequencesForGene(Sequences, gene)
            CovMat = tools.StackData(Sequences_per_gene, add = 'all')
            CovMat = GenerateFeatures(Ix, CovMat)
            SampleSame.append(CovMat[:, np.sum(CovMat, axis = 0) > 0])
            del CovMat
        print '\n'
        #Create X
        X = np.concatenate(SampleSame + SampleOther, axis =1)
        #Create Y
        Y0 = np.zeros((1, np.sum([Mat.shape[1] for Mat in SampleSame])), dtype=np.int)
        Y1 = np.ones((1, np.sum([Mat.shape[1] for Mat in SampleOther])), dtype=np.int)
        Y = np.hstack((Y0, Y1))[0,:]
        
        Cs = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5]
        LR = LogisticRegressionCV(Cs = Cs, penalty='l2', tol=0.01, class_weight='auto')
        LR.fit(X.T, Y.T)
        NewTransitionParametersLogReg[CurrState] = LR
        print 'Elapsed time: ' + str(time.time() - t)
        del Ix1, Ix2, Ix, SampleSame, SampleOther       

    return NewTransitionParametersLogReg


def FitTransistionParametersUnif(Sequences, Background, TransitionParameters, CurrPath, C):
    '''
    This function determines the optimal parameters of the logistic regression for predicting the TransitionParameters
    '''

    #Generate features from the CurrPaths and the Information in the coverage
    TransitionMatrix = TransitionParameters[0]
    NewTransitionParametersLogReg = {}
    t = time.time()

    #Iterate over the possible transistions
    assert (TransitionMatrix.shape[0] > 1), 'Only two states are currently allowed'
    CurrClass = 0

    genes = CurrPath.keys()
    genes = random.sample(genes, min(len(genes), 1000))

    NrOfStates = TransitionMatrix.shape[0]
    for CurrState in range(NrOfStates):
        CurrClass = 0
        Xs = []
        Ys = []
        print "Learning transistion model for State " + str(CurrState)
        
        SampleSame = []
        SampleOther = []
        #Iterate over the genes
        print 'Loading data'
        for i, gene in enumerate(genes):
            if i % 1000 == 0:
                sys.stdout.write('.')
                sys.stdout.flush()
            #Positions where the path is in the current state
            Ix1 = CurrPath[gene][:-1] == CurrState 
            #Positions where the subsequent position path is in the "next" state
            Ix2 = CurrPath[gene][1:] == CurrState
            #Positions where the path changes from the current state to the other state
            Ix = np.where(Ix1 * Ix2)[0]
            #CovMat = Sequences[gene]['CovNr'].toarray()
            Sequences_per_gene = PreloadSequencesForGene(Sequences, gene)
            CovMat = tools.StackData(Sequences_per_gene, add = 'all')
            nr_of_samples = CovMat.shape[0]
            CovMat[CovMat < 0] = 0
            if np.sum(np.sum(np.isnan(CovMat)))> 0:
                pdb.set_trace()
            CovMat = GenerateFeatures(Ix, CovMat)
            if np.sum(np.sum(np.isnan(CovMat)))> 0 or np.sum(np.sum(np.isinf(CovMat)))> 0:
                pdb.set_trace()
            if CovMat.shape[1] == 0:
                CovMat = np.zeros((nr_of_samples, 1))
                SampleSame.append(CovMat)
            else:
                SampleSame.append(CovMat)
            del CovMat
            Ix = np.where((Ix1 * Ix2) == 0)[0]
            CovMat = tools.StackData(Sequences_per_gene, add = 'all')
            CovMat[CovMat < 0] = 0
            if np.sum(np.sum(np.isnan(CovMat)))> 0:
                pdb.set_trace()
            CovMat = GenerateFeatures(Ix, CovMat)
            if np.sum(np.sum(np.isnan(CovMat)))> 0 or np.sum(np.sum(np.isinf(CovMat)))> 0:
                pdb.set_trace()
            if CovMat.shape[1] == 0:
                CovMat = np.zeros((nr_of_samples, 1))
                SampleOther.append(CovMat)
            else:
                SampleOther.append(CovMat)
            del CovMat
            
        print '\n'
        #Create X
    X_P = np.concatenate(SampleSame, axis =1)
    X_N = np.concatenate(SampleOther, axis =1)
    X = np.hstack((X_P, X_N))
    Y = np.hstack((np.ones((1, X_P.shape[1])), np.zeros((1, X_N.shape[1]))))[0,:]
    del X_P, X_N
    n_iter = max(5, np.ceil(10**6 / len(Y)))

    LR = SGDClassifier(loss="log", n_iter = n_iter).fit(X.T, Y.T)

    NewTransitionParametersLogReg = LR
    del Ix1, Ix2,  Ix, SampleSame, SampleOther, X, Y, Xs, Ys 
    print 'Done: Elapsed time: ' + str(time.time() - t)

    return NewTransitionParametersLogReg


def FitTransistionParametersUnifBck(Sequences, Background, TransitionParameters, CurrPath, C):
    '''
    This function determines the optimal parameters of the logistic regression for predicting the TransitionParameters
    '''

    #Generate features from the CurrPaths and the Information in the coverage
    TransitionMatrix = TransitionParameters[0]
    NewTransitionParametersLogReg = {}
    t = time.time()
    #Iterate over the possible transistions
    assert (TransitionMatrix.shape[0] > 1), 'Only two states are currently allowed'
    CurrClass = 0

    genes = CurrPath.keys()
    genes = random.sample(genes, min(len(genes), 1000))

    NrOfStates = TransitionMatrix.shape[0]
    for CurrState in range(NrOfStates):
        CurrClass = 0
        Xs = []
        Ys = []
        print "Learning transistion model for State " + str(CurrState)
        
        SampleSame = []
        SampleOther = []
        #Iterate over the genes
        print 'Loading data'
        for i, gene in enumerate(genes):
            if i % 1000 == 0:
                sys.stdout.write('.')
                sys.stdout.flush()
            #Positions where the path is in the current state
            Ix1 = CurrPath[gene][:-1] == CurrState 
            #Positions where the subsequent position path is in the "next" state
            Ix2 = CurrPath[gene][1:] == CurrState
            #Positions where the path changes from the current state to the other state
            Ix = np.where(Ix1 * Ix2)[0]

            Sequences_per_gene = PreloadSequencesForGene(Sequences, gene)
            Background_per_gene = PreloadSequencesForGene(Background, gene)

            CovMatSeq = tools.StackData(Sequences_per_gene, add = 'all')
            CovMatBck = tools.StackData(Background_per_gene, add = 'only_cov')
            CovMat = np.vstack((CovMatSeq, CovMatBck))
            nr_of_samples = CovMat.shape[0]
            CovMat[CovMat < 0] = 0
            if np.sum(np.sum(np.isnan(CovMat)))> 0:
                pdb.set_trace()
            CovMat = GenerateFeatures(Ix, CovMat)
            if np.sum(np.sum(np.isnan(CovMat)))> 0 or np.sum(np.sum(np.isinf(CovMat)))> 0:
                pdb.set_trace()
            if CovMat.shape[1] == 0:
                CovMat = np.zeros((nr_of_samples, 1))
                SampleSame.append(CovMat)
            else:
                SampleSame.append(CovMat)
            del CovMat
            Ix = np.where((Ix1 * Ix2) == 0)[0]
            CovMatSeq = tools.StackData(Sequences_per_gene, add = 'all')
            CovMatBck = tools.StackData(Background_per_gene, add = 'only_cov')
            CovMat = np.vstack((CovMatSeq, CovMatBck))
            CovMat[CovMat < 0] = 0
            if np.sum(np.sum(np.isnan(CovMat)))> 0:
                pdb.set_trace()
            CovMat = GenerateFeatures(Ix, CovMat)
            if np.sum(np.sum(np.isnan(CovMat)))> 0 or np.sum(np.sum(np.isinf(CovMat)))> 0:
                pdb.set_trace()
            if CovMat.shape[1] == 0:
                CovMat = np.zeros((nr_of_samples, 1))
                SampleOther.append(CovMat)
            else:
                SampleOther.append(CovMat)
            del CovMat
            
        print '\n'
        #Create X
    X_P = np.concatenate(SampleSame, axis =1)
    X_N = np.concatenate(SampleOther, axis =1)
    X = np.hstack((X_P, X_N))
    Y = np.hstack((np.ones((1, X_P.shape[1])), np.zeros((1, X_N.shape[1]))))[0,:]
    del X_P, X_N
    n_iter = max(5, np.ceil(10**6 / len(Y)))

    LR = SGDClassifier(loss="log", n_iter = n_iter).fit(X.T, Y.T)

    NewTransitionParametersLogReg = LR
    del Ix1, Ix2,  Ix, SampleSame, SampleOther, X, Y, Xs, Ys 
    print 'Done: Elapsed time: ' + str(time.time() - t)

    return NewTransitionParametersLogReg


def FitTransistionParametersMultinomialSeparate(Sequences, Background, TransitionParameters, CurrPath, C):
    '''
    This function determines the optimal parameters of the logistic regression for predicting the TransitionParameters
    '''

    #Generate features from the CurrPaths and the Information in the coverage
    TransitionMatrix = TransitionParameters[0]
    NewTransitionParametersLogReg = {}
    t = time.time()
    #Iterate over the possible transistions
    assert (TransitionMatrix.shape[0] > 1), 'Only two states are currently allowed'
    CurrClass = 0

    genes = CurrPath.keys()
    genes = random.sample(genes, min(len(genes), 1000))

    NrOfStates = TransitionMatrix.shape[0]
    for CurrState in range(NrOfStates):
        CurrClass = 0
        Xs = []
        Ys = []
        print "Learning transistion model for State " + str(CurrState)
        for NextState in range(NrOfStates):
            SampleSame = []
            SampleOther = []
            #Iterate over the genes
            print 'Loading data'
            for i, gene in enumerate(genes):
                if i % 1000 == 0:
                    sys.stdout.write('.')
                    sys.stdout.flush()
                #Positions where the path is in the current state
                Ix1 = CurrPath[gene][:-1] == CurrState 
                #Positions where the subsequent position path is in the "next" state
                Ix2 = CurrPath[gene][1:] == NextState
                #Positions where the path changes from the current state to the other state
                Ix = np.where(Ix1 * Ix2)[0]
                Sequences_per_gene = tools.PreloadSequencesForGene(Sequences, gene)
                CovMat = tools.StackData(Sequences_per_gene, add = 'all')
                CovMat[CovMat < 0] = 0
                if np.sum(np.sum(np.isnan(CovMat)))> 0:
                    pdb.set_trace()
                CovMat = GenerateFeatures(Ix, CovMat)
                if np.sum(np.sum(np.isnan(CovMat)))> 0 or np.sum(np.sum(np.isinf(CovMat)))> 0:
                    pdb.set_trace()
                if CovMat.shape[1] == 0:
                    CovMat = np.zeros((2, 1))
                    SampleOther.append(CovMat)
                else:
                    SampleOther.append(CovMat)
                del CovMat
            print '\n'
                #Create X
            X = np.concatenate(SampleOther, axis =1)
            #Create Y
            Y = (np.ones((1, np.sum([Mat.shape[1] for Mat in SampleOther])), dtype=np.int) * CurrClass)[0,:]
            Xs.append(X)
            Ys.append(Y)
            CurrClass += 1

        X = np.concatenate(Xs, axis = 1)
        Y = np.concatenate(Ys)
        n_iter = max(5, np.ceil(10**6 / len(Y)))
        LR = SGDClassifier(loss="log", n_iter = n_iter).fit(X.T, Y.T)

        NewTransitionParametersLogReg[CurrState] = LR
        del Ix1, Ix2,  Ix, SampleSame, SampleOther, X, Y, Xs, Ys 
    print 'Done: Elapsed time: ' + str(time.time() - t)

    return NewTransitionParametersLogReg


def FitTransistionParametersSimple(Sequences, Background, TransitionParameters, CurrPath, C):
    '''
    This function determines the optimal parameters of the logistic regression for predicting the TransitionParameters
    '''

    #Generate features from the CurrPaths and the Information in the coverage
    TransitionMatrix = TransitionParameters[0]
    NewTransitionParametersLogReg = {}
    t = time.time()
    #Iterate over the possible transistions
    assert (TransitionMatrix.shape[0] > 1), 'Only two states are currently allowed'

    genes = CurrPath.keys()
    genes = random.sample(genes, min(len(genes), 1000))

    NrOfStates = TransitionMatrix.shape[0]
    Xs = []
    Ys = []
    SampleSame = []
    SampleOther = []
    print "Learning transistion model"
    print "Iterating over genes"
    print 'Fitting transistion parameters: I'
    print 'Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
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
        
    print 'Fitting transistion parameters: II'
    print 'Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    len_same = np.sum([Mat.shape[1] for Mat in SampleSame])
    len_other = np.sum([Mat.shape[1] for Mat in SampleOther])

    X = np.concatenate(SampleSame + SampleOther, axis =1)
    del SampleSame, SampleOther

    #Create Y
    Y = np.hstack((np.ones((1, len_same), dtype=np.int), np.zeros((1, len_other), dtype=np.int)))[0,:]
    print 'Fitting transistion parameters: III'
    print 'Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    n_iter = max(5, np.ceil(10**6 / len(Y)))
    NewTransitionParametersLogReg = SGDClassifier(loss="log", n_iter = n_iter).fit(X.T, Y.T)
    
    print 'Fitting transistion parameters: IV'
    print 'Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    del Ix1, Ix2,  Ix, X, Y, Xs, Ys 
    print 'Done: Elapsed time: ' + str(time.time() - t)
    print 'Fitting transistion parameters: V'
    print 'Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    return NewTransitionParametersLogReg


#@profile
def FitTransistionParametersSimpleBck(Sequences, Background, TransitionParameters, CurrPath, C):
    '''
    This function determines the optimal parameters of the logistic regression for predicting the TransitionParameters
    '''
    #Generate features from the CurrPaths and the Information in the coverage
    TransitionMatrix = TransitionParameters[0]
    NewTransitionParametersLogReg = {}

    t = time.time()
    #Iterate over the possible transistions
    assert (TransitionMatrix.shape[0] > 1), 'Only two states are currently allowed'

    genes = CurrPath.keys()
    genes = random.sample(genes, min(len(genes), 1000))

    NrOfStates = TransitionMatrix.shape[0]
    Xs = []
    Ys = []
    SampleSame = []
    SampleOther = []
    print "Learning transistion model"
    print "Iterating over genes"

    for i, gene in enumerate(genes):
        if i % 1000 == 0:
            sys.stdout.write('.')
            sys.stdout.flush()
        #Get data
        Sequences_per_gene = tools.PreloadSequencesForGene(Sequences, gene)
        Background_per_gene = tools.PreloadSequencesForGene(Background, gene)

        CovMatSeq = tools.StackData(Sequences_per_gene, add = 'all')
        CovMatBck = tools.StackData(Background_per_gene, add = 'only_cov')
        CovMat = np.vstack((CovMatSeq, CovMatBck))
        nr_of_samples = CovMat.shape[0]
        CovMat[CovMat < 0] = 0
        for CurrState in range(NrOfStates):
            for NextState in range(NrOfStates):
                #Positions where the path is in the current state
                Ix1 = CurrPath[gene][:-1] == CurrState 
                #Positions where the subsequent position path is in the "next" state
                Ix2 = CurrPath[gene][1:] == NextState
                #Positions where the path changes from the current state to the other state
                Ix = np.where(Ix1 * Ix2)[0]
                
                if np.sum(np.sum(np.isnan(CovMat))) > 0:
                    continue
                CovMatIx = GenerateFeatures(Ix, CovMat)
                if np.sum(np.sum(np.isnan(CovMatIx))) > 0 or np.sum(np.sum(np.isinf(CovMatIx))) > 0:
                    continue
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
        del CovMat, CovMatIx, CovMatSeq, CovMatBck
    X = np.concatenate(SampleSame + SampleOther, axis =1)
    #Create Y 
    len_same = np.sum([Mat.shape[1] for Mat in SampleSame])
    len_other = np.sum([Mat.shape[1] for Mat in SampleOther])
    Y = np.hstack((np.ones((1, len_same), dtype=np.int), np.zeros((1, len_other), dtype=np.int)))[0,:]
    
    n_iter = max(5, np.ceil(10**6 / len(Y)))
    NewTransitionParametersLogReg = SGDClassifier(loss="log", n_iter = n_iter).fit(X.T, Y.T)

    del Ix1, Ix2,  Ix, SampleSame, SampleOther, X, Y, Xs, Ys 

    print 'Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print 'Done: Elapsed time: ' + str(time.time() - t)

    return NewTransitionParametersLogReg


def FitTransistionParametersMultinomialSeparateManual(Sequences, Background, TransitionParameters, CurrPath, C):
    '''
    This function determines the optimal parameters of the logistic regression for predicting the TransitionParameters
    '''

    #Generate features from the CurrPaths and the Information in the coverage
    TransitionMatrix = TransitionParameters[0]
    NewTransitionParametersLogReg = {}
    t = time.time()
    #Iterate over the possible transistions
    assert (TransitionMatrix.shape[0] > 1), 'Only two states are currently allowed'
    CurrClass = 0
    
    NrOfStates = TransitionMatrix.shape[0]
    for CurrState in range(NrOfStates):
        CurrClass = 0
        Xs = []
        Ys = []
        print "Learning transistion model for State " + str(CurrState)
        for NextState in range(NrOfStates):
            SampleSame = []
            SampleOther = []
            #Iterate over the genes
            print 'Loading data'
            for i, gene in enumerate(CurrPath.keys()):
                if i % 1000 == 0:
                    sys.stdout.write('.')
                    sys.stdout.flush()
                #Positions where the path is in the current state
                Ix1 = CurrPath[gene][:-1] == CurrState 
                #Positions where the subsequent position path is in the "next" state
                Ix2 = CurrPath[gene][1:] == NextState
                #Positions where the path changes from the current state to the other state
                Ix = np.where(Ix1 * Ix2)[0]
                CovMat = Sequences[gene]['CovNr'].toarray() + Sequences[gene]['ConvNrTC'].toarray() + Sequences[gene]['ConvNrNonTC'].toarray()
                CovMat = GenerateFeatures(Ix, CovMat)
                CovMat = GenerateFeatures(Ix, CovMat)
                if CovMat.shape[1] == 0:
                    CovMat = np.zeros((2, 1))
                    SampleOther.append(CovMat)
                else:
                    SampleOther.append(CovMat)
                
                del CovMat
            print '\n'
                #Create X
            X = np.concatenate(SampleOther, axis =1)
            #Create Y
            Y = (np.ones((1, np.sum([Mat.shape[1] for Mat in SampleOther])), dtype=np.int) * CurrClass)[0,:]
            Xs.append(X)
            Ys.append(Y)
            CurrClass += 1

        X = np.concatenate(Xs, axis = 1)
        Y = np.concatenate(Ys, axis = 1)
        
        LR = LogisticRegression(C = 1, penalty='l2', tol=0.01, solver='lbfgs', multi_class='multinomial')
        LR.fit(X.T, Y.T)
        NewTransitionParametersLogReg[CurrState] = [LR, np.unique(Y)]
        del Ix1, Ix2,  Ix, SampleSame, SampleOther, X, Y, Xs, Ys 
    print 'Done: Elapsed time: ' + str(time.time() - t)

    return NewTransitionParametersLogReg


def FitTransistionParametersMultinomial(Sequences, Background, TransitionParameters, CurrPath, C):
    '''
    This function determines the optimal parameters of the logistic regression for predicting the TransitionParameters
    '''

    #Generate features from the CurrPaths and the Information in the coverage
    TransitionMatrix = TransitionParameters[0]
    t = time.time()
    #Iterate over the possible transistions
    assert (TransitionMatrix.shape[0] > 1), 'Only two states are currently allowed'
    CurrClass = 0
    Xs = []
    Ys = []
    NrOfStates = TransitionMatrix.shape[0]
    for CurrState in range(NrOfStates):
        print "Learning transistion model for State " + str(CurrState)
        for NextState in range(NrOfStates):
            SampleSame = []
            SampleOther = []
            #Iterate over the genes
            print 'Loading data'
            for i, gene in enumerate(CurrPath.keys()):
                if i % 1000 == 0:
                    sys.stdout.write('.')
                    sys.stdout.flush()
                #Positions where the path is in the current state
                Ix1 = CurrPath[gene][:-1] == CurrState 
                #Positions where the subsequent position path is in the "next" state
                Ix2 = CurrPath[gene][1:] == NextState
                #Positions where the path changes from the current state to the other state
                Ix = np.where(Ix1 * Ix2)[0]
                CovMat = Sequences[gene]['CovNr'].toarray()  + Sequences[gene]['ConvNrTC'].toarray() + Sequences[gene]['ConvNrNonTC'].toarray()
                CovMat = GenerateFeatures(Ix, CovMat)
                if CovMat.shape[1] == 0:
                    CovMat = np.zeros((2, 1))
                    SampleOther.append(CovMat)
                else:
                    SampleOther.append(CovMat)
                del CovMat
            print '\n'
                #Create X
            X = np.concatenate(SampleOther, axis =1)
            #Create Y
            Y = (np.ones((1, np.sum([Mat.shape[1] for Mat in SampleOther])), dtype=np.int) * CurrClass)[0,:]
            CurrClass += 1
            Xs.append(X)
            Ys.append(Y)

    #pdb.set_trace()
    X = np.concatenate(Xs, axis = 1)
    Y = np.concatenate(Ys, axis = 1)
    LR = LogisticRegression(C = 1, penalty='l2', tol=0.1, solver='lbfgs', multi_class='multinomial', verbose = 1, max_iter=1000)
    
    n_iter = max(5, np.ceil(10**6 / len(Y)))
    LR = SGDClassifier(loss="log", n_iter = n_iter).fit(X.T, Y.T)
    
    NewTransitionParametersLogReg = LR
    print 'Done: Elapsed time: ' + str(time.time() - t)
    del Ix1, Ix2,  Ix, SampleSame, SampleOther, X, Y, Xs, Ys 

    return NewTransitionParametersLogReg



def GenerateFeatures(Ix, CovMat):
    '''
    This funnction generates, for a set of positions, the features for the logistic regression from the Coverage matrix
    '''

    FeatureMatrix = np.log(1 + CovMat[:, Ix])
    return FeatureMatrix
