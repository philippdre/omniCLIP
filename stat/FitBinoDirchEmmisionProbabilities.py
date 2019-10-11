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
sys.path.append('./Utils/')
from scipy.special import psi
from scipy.stats import nbinom
import multdirichletVect
import numpy as np


#@profile 
def ComputePrior(*args):
    '''
    This function computes the prior for the dirichlet model
    '''

    alpha, OldPriorMatrix, Counts, NrOfCounts, EmissionParameters = args
    #Prepare the return variable
    NrOfNegObs = 0
    NrOfPosObs = 0
    NrOfBackObs = 0

    PriorMatrix = np.zeros_like(OldPriorMatrix)
    #Iterate over the states:
    for State in list(Counts.keys()):
        #Computet the likelihood for the current observation for both models

        #State 0 
        p_0 = EmissionParameters['ExpNonTC'][0][0]
        p_1 = EmissionParameters['ExpNonTC'][1][0]
        n_0 = EmissionParameters['ExpNonTC'][0][1]
        n_1 = EmissionParameters['ExpNonTC'][1][1]
        ExprLikelihood = nbinom.logpmf(np.sum(Counts[State][list(range(0,6,2)), :], axis = 0), n_0, p_0) + nbinom.logpmf(np.sum(Counts[State][list(range(1,6,2)), :], axis = 0), n_1, p_1)
        RatioLikelihood = (multdirichletVect.log_pdf_vect(Counts[State][list(range(0,6,2)), :], alpha[3:6]) + multdirichletVect.log_pdf_vect(Counts[State][list(range(1,6,2)), :], alpha[3:6]))
        CurrLogLikelihoodNeg = (ExprLikelihood + RatioLikelihood)

        #State 1
        p_0 = EmissionParameters['ExpTC'][0][0]
        p_1 = EmissionParameters['ExpTC'][1][0]
        n_0 = EmissionParameters['ExpTC'][0][1]
        n_1 = EmissionParameters['ExpTC'][1][1]
        ExprLikelihood = nbinom.logpmf(np.sum(Counts[State][list(range(0,6,2)), :], axis = 0), n_0, p_0) + nbinom.logpmf(np.sum(Counts[State][list(range(1,6,2)), :], axis = 0), n_1, p_1)
        RatioLikelihood = (multdirichletVect.TwoBinomlog_pdf_vect(Counts[State][list(range(0,6,2)), :], Counts[State][list(range(1,6,2)), :], alpha[0:3]))
        CurrLogLikelihoodPos = (ExprLikelihood + RatioLikelihood)

        #State 2
        p_0 = EmissionParameters['ExpBck'][0][0]
        p_1 = EmissionParameters['ExpBck'][1][0]
        n_0 = EmissionParameters['ExpBck'][0][1]
        n_1 = EmissionParameters['ExpBck'][1][1]
        ExprLikelihood = nbinom.logpmf(np.sum(Counts[State][list(range(0,6,2)), :], axis = 0), n_0, p_0) + nbinom.logpmf(np.sum(Counts[State][list(range(1,6,2)), :], axis = 0), n_1, p_1)
        RatioLikelihood = (multdirichletVect.log_pdf_vect(Counts[State][list(range(0,6,2)), :], alpha[6:9]) + multdirichletVect.log_pdf_vect(Counts[State][list(range(1,6,2)), :], alpha[6:9]))
        CurrLogLikelihoodBack = (ExprLikelihood + RatioLikelihood)

        #Determine the likelihodd that are not inf 
        IxNonInf = np.isinf(CurrLogLikelihoodNeg * CurrLogLikelihoodPos * CurrLogLikelihoodBack) == 0
        CurrLogLikelihoodNeg = CurrLogLikelihoodNeg[IxNonInf]
        CurrLogLikelihoodPos = CurrLogLikelihoodPos[IxNonInf]
        CurrLogLikelihoodBack = CurrLogLikelihoodBack[IxNonInf]
        CurrCounts = NrOfCounts[State][IxNonInf]

        #Determine the probabilty of being in each state
        TempSumPos = (np.exp(CurrLogLikelihoodPos) /(np.exp(CurrLogLikelihoodPos) + np.exp(CurrLogLikelihoodNeg + np.exp(CurrLogLikelihoodBack))))
        TempSumNeg = (np.exp(CurrLogLikelihoodNeg) /(np.exp(CurrLogLikelihoodPos) + np.exp(CurrLogLikelihoodNeg + np.exp(CurrLogLikelihoodBack))))
        CurrCounts = CurrCounts[np.isnan(TempSumPos + TempSumNeg) == 0]
        Ix = np.isnan(TempSumPos + TempSumNeg) == 0
        TempSumPos = TempSumPos[Ix]
        TempSumNeg = TempSumNeg[Ix] 
        NrOfPosObs += np.sum(TempSumPos * CurrCounts)
        NrOfNegObs += np.sum(TempSumNeg * CurrCounts)
        NrOfBackObs += np.sum((1 - (TempSumPos + TempSumNeg)) * CurrCounts)

    #Compute the new prior
    PriorMatrix[0, 0] = NrOfNegObs / (NrOfPosObs + NrOfNegObs + NrOfBackObs)
    PriorMatrix[1, 0] = NrOfPosObs / (NrOfPosObs + NrOfNegObs + NrOfBackObs)
    PriorMatrix[2, 0] = NrOfBackObs / (NrOfPosObs + NrOfNegObs + NrOfBackObs)

    return PriorMatrix

#@profile 
def ComputeStateProbForGeneMD_unif(*args):
    '''
    This function computes the prior for the dirichlet model
    '''
    Counts, alpha, State, EmissionParameters = args

    tracks_per_rep = alpha.shape[0]

    NrOfReplicates = Counts.shape[0] / tracks_per_rep

    Prob = np.zeros((Counts.shape[1]))

    #Precompute the Zero values
    IxZeros = np.sum(Counts, axis = 0) == 0
    IxNonZeros = np.sum(Counts, axis = 0) > 0

    #If there are zero values, add them to Counts in the first column
    if np.sum(IxZeros) > 0:
        ZeroCounts = np.zeros((Counts.shape[0], 1))
        Counts = np.hstack((ZeroCounts, Counts[:, IxNonZeros]))
    else:
        Counts  = Counts[:, IxNonZeros]

    RatioLikelihood = multdirichletVect.log_pdf_vect(Counts[0 : tracks_per_rep, :], alpha)
    for i in range(1, NrOfReplicates):
        RatioLikelihood += multdirichletVect.log_pdf_vect(Counts[i * tracks_per_rep:(i + 1) * tracks_per_rep, :], alpha)

    #Computet the likelihood for the current observation for the current model models
    if np.sum(IxZeros) > 0:
        Prob[IxNonZeros] =  RatioLikelihood[0, 1:]
        Prob[IxZeros] = np.tile(RatioLikelihood[0, 0] , (1, np.sum(IxZeros)))
    else:
        Prob = RatioLikelihood[0, :]

    return Prob

##@profile
#@profile 
def ComputeStateProbForGeneMD_unif_rep(*args):
    '''
    This function computes the prior for the dirichlet model
    '''

    Counts, alpha, State, EmissionParameters = args
    
    tracks_per_rep = alpha.shape[0]
    NrOfReplicates = Counts.shape[0] / tracks_per_rep


    Prob = np.zeros((Counts.shape[1]))

    #Precompute the Zero values
    IxZeros = np.sum(Counts, axis = 0) == 0
    IxNonZeros = np.sum(Counts, axis = 0) > 0

    #If there are zero values, add them to Counts in the first column
    if np.sum(IxZeros) > 0:
        ZeroCounts = np.zeros((Counts.shape[0], 1))
        Counts = np.hstack((ZeroCounts, Counts[:, IxNonZeros]))
    else:
        Counts  = Counts[:, IxNonZeros]

    RatioLikelihood = multdirichletVect.log_pdf_vect_rep(Counts, alpha, tracks_per_rep, NrOfReplicates)
    #Computet the likelihood for the current observation for the current model models

    if np.sum(IxZeros) > 0:
        Prob[IxNonZeros] =  RatioLikelihood[0, 1:]

        #if len(Prob[IxZeros].shape) ==1:
        Prob[IxZeros] = RatioLikelihood[0, 0]
        #else:
        #Prob[IxZeros] = np.tile(RatioLikelihood[0, 0] , (1, np.sum(IxZeros)))
    else:
        Prob = RatioLikelihood[0, :]

    return Prob

##@profile
#@profile 
def MDK_f_joint_vect_unif(x, *args):
    '''
    This function computes the lieklihood of the parameters
    '''

    alpha = x
    Counts, NrOfCounts, EmissionParameters = args
    #Prepare the return variable
    LogLikelihood = 0.0    
    #NrOfReplicates = EmissionParameters['NrOfReplicates']
    NrOfReplicates = Counts.shape[0] / alpha.shape[0]
    tracks_per_rep =  x.shape[0]

    RatioLikelihood = multdirichletVect.log_pdf_vect_rep(Counts, alpha, tracks_per_rep, NrOfReplicates)
    CurrLogLikelihood = RatioLikelihood * NrOfCounts
    LogLikelihood = np.sum(CurrLogLikelihood[np.isinf(CurrLogLikelihood) == 0])

    return -LogLikelihood


##@profile
#@profile 
def MDK_f_prime_joint_vect_unif(x, *args):
    Counts, NrOfCounts, EmissionParameters = args
    
    tracks_per_rep = x.shape[0]
    NrOfReplicates = Counts.shape[0] / tracks_per_rep
    #Prepare the return variable
    LogLikelihood = np.zeros_like(x, dtype=np.float)

    #compute the likelihood
    curr_alpha = x

    k = Counts[0 : tracks_per_rep, :]
    k, Ks = multdirichletVect.expand_k(k)
    for rep in range(1, int(NrOfReplicates)):
        new_k, Ks = multdirichletVect.expand_k(Counts[rep * tracks_per_rep:(rep + 1) * tracks_per_rep, :])
        k += new_k

    DBase = psi(np.sum(curr_alpha)) - psi(np.sum(k, axis=0) + np.sum(curr_alpha)) 

    for J in range(0, curr_alpha.shape[0]):
        D = psi(curr_alpha[J] + k) -  psi(curr_alpha[J])

    CurrLogLikeliehood = np.float64((D + DBase) * NrOfCounts)
    LogLikelihood[J] += np.sum(CurrLogLikeliehood[np.isinf(CurrLogLikeliehood) == 0])

    return -LogLikelihood



##@profile
#@profile 
def MD_f_joint_vect_unif(x, *args):
    '''
    This function computes the lieklihood of the parameters
    '''

    alpha = x
    Counts, NrOfCounts, EmissionParameters = args

    #Prepare the return variable
    LogLikelihood = 0.0    
    
    tracks_per_rep =  x.shape[0]
    NrOfReplicates = Counts.shape[0] / tracks_per_rep

    RatioLikelihood = multdirichletVect.log_pdf_vect(Counts[0 : tracks_per_rep, :], alpha)
    for i in range(1, int(NrOfReplicates)):
        RatioLikelihood += multdirichletVect.log_pdf_vect(Counts[i * tracks_per_rep:(i + 1) * tracks_per_rep, :], alpha)
    CurrLogLikelihood = RatioLikelihood * np.float64(NrOfCounts)
    LogLikelihood += np.sum(CurrLogLikelihood[np.isinf(CurrLogLikelihood) == 0])

    return -LogLikelihood


##@profile
#@profile 
def MD_f_prime_joint_vect_unif(x, *args):
    Counts, NrOfCounts, EmissionParameters = args
    
    tracks_per_rep = x.shape[0]
    NrOfReplicates = Counts.shape[0] / tracks_per_rep

    #Prepare the return variable
    LogLikelihood = np.zeros_like(x, dtype=np.float)

    #compute the likelihood
    curr_k = Counts[0 : tracks_per_rep, :]
    curr_alpha = x

    DBase = NrOfReplicates * psi(np.sum(curr_alpha)) - psi(np.sum(curr_k, axis=0) + np.sum(curr_alpha)) 
    for rep in range(1, int(NrOfReplicates)):
        curr_k = Counts[rep * tracks_per_rep:(rep + 1) * tracks_per_rep, :]
        DBase -=  psi(np.sum(curr_k, axis=0) + np.sum(curr_alpha))  

    for J in range(0, curr_alpha.shape[0]):
        curr_k = Counts[J, :]
        ix_zero = curr_k == 0

        D = np.float64(np.zeros_like(curr_k))
        if np.isscalar(D):
            D = psi(curr_alpha[J] + curr_k) -  psi(curr_alpha[J])
        else:
            D[ix_zero] = psi(curr_alpha[J] + curr_k[ix_zero]) -  psi(curr_alpha[J])

        for rep in range(1, int(NrOfReplicates)):
            curr_k = Counts[rep * tracks_per_rep + J, :]
            ix_zero = curr_k == 0
            if np.isscalar(D):
                D += psi(curr_alpha[J] + curr_k)  - psi(curr_alpha[J])
            else:
                D[ix_zero] += psi(curr_alpha[J] + curr_k[ix_zero])  - psi(curr_alpha[J])
            
        CurrLogLikeliehood = np.float64((D + DBase) * np.float64(NrOfCounts))
        LogLikelihood[J] += np.sum(CurrLogLikeliehood[np.isinf(CurrLogLikeliehood) == 0])
    return -LogLikelihood
