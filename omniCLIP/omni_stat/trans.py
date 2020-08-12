"""
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
"""

import numpy as np
import random
from scipy.special import logsumexp
from sklearn.linear_model import SGDClassifier
import sys
import time

from omniCLIP.data_parsing import tools
from omniCLIP.omni_stat.utils import get_mem_usage


def PredictTransistions(Counts, TransitionParameters, NrOfStates, Type='multi',
                        verbosity=1):
    """Predict the transition probabilities for a gene."""
    return PredictTransistionsSimple(Counts, TransitionParameters, NrOfStates)


def PredictTransistionsSimple(Counts, TransitionParameters, NrOfStates,
                              verbosity=1):
    """Predict the transition probabilities for a gene."""
    TransitionParametersLogReg = TransitionParameters[1]
    TransistionProb = (np.ones((NrOfStates, NrOfStates, Counts.shape[1]))
                       * np.log((1 / np.float64(NrOfStates))))

    # Genererate the features
    CovMat = GenerateFeatures(
        np.array(list(range(Counts.shape[1] - 1))), Counts)

    ix_nonzero = np.sum(CovMat, axis=0) > 0
    # Create the probabilities
    TempProb = TransitionParametersLogReg.predict_log_proba(CovMat.T).T

    NormFactor = np.ones((TempProb.shape[1]))
    for CurrentState in range(NrOfStates):
        for NextState in range(NrOfStates):
            if CurrentState == NextState:
                TransistionProb[NextState, CurrentState, 1:] = TempProb[1, :]
            else:
                TransistionProb[NextState, CurrentState, 1:] = TempProb[0, :]

        # Normalize the transition probabilities

        if CurrentState == 0:
            # Only compute the normalizing factor once as compuation is
            # expensive
            if np.sum(ix_nonzero) > 0:
                # Nonzero entries
                NormFactor[ix_nonzero] = logsumexp(
                    TransistionProb[:, CurrentState, 1:][:, ix_nonzero],
                    axis=0)
            if np.sum(ix_nonzero == 0) > 0:
                # Zero entries
                first_zero_pos = np.where(ix_nonzero == 0)[0][0]
                NormFactor[ix_nonzero == 0] = logsumexp(
                    TransistionProb[:, CurrentState, 1:][:, first_zero_pos],
                    axis=0)

        for NextState in range(NrOfStates):
            TransistionProb[NextState, CurrentState, 1:] -= NormFactor

    del TempProb
    return TransistionProb


def FitTransistionParameters(Sequences, Background, TransitionParameters,
                             CurrPath, Type='multi', verbosity=1):
    """Determine optimal logistic regression parameters.

    Return the optimal parameters of the logistic regression for predicting
    the TransitionParameters.
    """
    print('Fitting transition parameters')
    get_mem_usage(verbosity)

    NewTransitionParametersLogReg = FitTransistionParametersSimple(
        Sequences, Background,
        TransitionParameters, CurrPath,
        verbosity=verbosity)

    get_mem_usage(verbosity)

    return NewTransitionParametersLogReg


def FitTransistionParametersSimple(Sequences, Background, TransitionParameters,
                                   CurrPath, verbosity=1):
    """Determine optimal logistic regression parameters.

    Return the optimal parameters of the logistic regression for predicting
    the TransitionParameters.
    """
    # Generate features from the CurrPaths and the Information in the coverage
    TransitionMatrix = TransitionParameters[0]
    NewTransitionParametersLogReg = {}
    t = time.time()

    # Iterate over the possible transitions
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
    get_mem_usage(verbosity, msg='Fitting transition parameters: I')

    for i, gene in enumerate(genes):
        if i % 1000 == 0:
            sys.stdout.write('.')
            sys.stdout.flush()
        # Get data
        Sequences_per_gene = tools.PreloadSequencesForGene(Sequences, gene)
        CovMat = tools.StackData(Sequences_per_gene, add='all')
        CovMat[CovMat < 0] = 0
        nr_of_samples = CovMat.shape[0]
        for CurrState in range(NrOfStates):
            for NextState in range(NrOfStates):
                # Positions where the path is in the current state
                Ix1 = CurrPath[gene][:-1] == CurrState
                # Positions where the subsequent position path is in the "next"
                # state
                Ix2 = CurrPath[gene][1:] == NextState
                # Positions where the path changes from the current state to
                # the other state
                Ix = np.where(Ix1 * Ix2)[0]

                CovMatIx = GenerateFeatures(Ix, CovMat)

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

    get_mem_usage(verbosity, msg='Fitting transition parameters: II')

    len_same = np.sum([Mat.shape[1] for Mat in SampleSame])
    len_other = np.sum([Mat.shape[1] for Mat in SampleOther])

    X = np.concatenate(SampleSame + SampleOther, axis=1).T
    del SampleSame, SampleOther

    # Create Y
    Y = np.hstack(
        (np.ones((1, len_same), dtype=np.int),
         np.zeros((1, len_other), dtype=np.int)))[0, :].T
    classes = np.unique(Y)

    get_mem_usage(verbosity, msg='Fitting transition parameters: III')
    n_iter = max(5, np.ceil(10**6 / Y.shape[0]))

    NewTransitionParametersLogReg = SGDClassifier(loss="log", max_iter=n_iter)
    ix_shuffle = np.arange(X.shape[0])
    for n in range(n_iter):
        np.random.shuffle(ix_shuffle)
        for batch_ix in np.array_split(ix_shuffle, 50):
            NewTransitionParametersLogReg.partial_fit(
                X[batch_ix, :], Y[batch_ix], classes=classes)

    del Ix1, Ix2,  Ix, X, Y, Xs, Ys
    get_mem_usage(verbosity, t=t, msg='Fitting transition parameters: IV')

    return NewTransitionParametersLogReg


def GenerateFeatures(Ix, CovMat):
    """Generate the coverage matrix features for the logistic regression."""
    FeatureMatrix = np.log(1 + CovMat[:, Ix])
    return FeatureMatrix
