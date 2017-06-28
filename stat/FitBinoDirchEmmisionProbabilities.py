import sys
import numpy as np
import pylab as plt
sys.path.append('./Utils/')
import multdirichletVect
from scipy.special import psi
from scipy.stats import bernoulli
from numpy.random import multinomial
from scipy.stats import poisson
from numpy.random import dirichlet
from scipy.stats import nbinom


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
    for State in Counts.keys():
        #State 0 
        p_0 = EmissionParameters['ExpNonTC'][0][0]
        p_1 = EmissionParameters['ExpNonTC'][1][0]
        n_0 = EmissionParameters['ExpNonTC'][0][1]
        n_1 = EmissionParameters['ExpNonTC'][1][1]
        ExprLikelihood = nbinom.logpmf(np.sum(Counts[State][range(0,6,2), :], axis = 0), n_0, p_0) + nbinom.logpmf(np.sum(Counts[State][range(1,6,2), :], axis = 0), n_1, p_1)
        RatioLikelihood = (multdirichletVect.log_pdf_vect(Counts[State][range(0,6,2), :], alpha[3:6]) + multdirichletVect.log_pdf_vect(Counts[State][range(1,6,2), :], alpha[3:6]))
        CurrLogLikelihoodNeg = (ExprLikelihood + RatioLikelihood)

        #State 1
        p_0 = EmissionParameters['ExpTC'][0][0]
        p_1 = EmissionParameters['ExpTC'][1][0]
        n_0 = EmissionParameters['ExpTC'][0][1]
        n_1 = EmissionParameters['ExpTC'][1][1]
        ExprLikelihood = nbinom.logpmf(np.sum(Counts[State][range(0,6,2), :], axis = 0), n_0, p_0) + nbinom.logpmf(np.sum(Counts[State][range(1,6,2), :], axis = 0), n_1, p_1)
        RatioLikelihood = (multdirichletVect.TwoBinomlog_pdf_vect(Counts[State][range(0,6,2), :], Counts[State][range(1,6,2), :], alpha[0:3]))
        CurrLogLikelihoodPos = (ExprLikelihood + RatioLikelihood)

        #State 2
        p_0 = EmissionParameters['ExpBck'][0][0]
        p_1 = EmissionParameters['ExpBck'][1][0]
        n_0 = EmissionParameters['ExpBck'][0][1]
        n_1 = EmissionParameters['ExpBck'][1][1]
        ExprLikelihood = nbinom.logpmf(np.sum(Counts[State][range(0,6,2), :], axis = 0), n_0, p_0) + nbinom.logpmf(np.sum(Counts[State][range(1,6,2), :], axis = 0), n_1, p_1)
        RatioLikelihood = (multdirichletVect.log_pdf_vect(Counts[State][range(0,6,2), :], alpha[6:9]) + multdirichletVect.log_pdf_vect(Counts[State][range(1,6,2), :], alpha[6:9]))
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
        Prob[IxZeros] = np.tile(RatioLikelihood[0, 0] , (1, np.sum(IxZeros)))
    else:
        Prob = RatioLikelihood[0, :]
    return Prob


def MDK_f_joint_vect_unif(x, *args):
    '''
    This function computes the lieklihood of the parameters
    '''
    alpha = x

    Counts, NrOfCounts, EmissionParameters = args
    #Prepare the return variable
    LogLikelihood = 0.0    
    NrOfReplicates = Counts.shape[0] / alpha.shape[0]
    tracks_per_rep =  x.shape[0]

    RatioLikelihood = multdirichletVect.log_pdf_vect_rep(Counts, alpha, tracks_per_rep, NrOfReplicates)
    CurrLogLikelihood = RatioLikelihood * NrOfCounts
    LogLikelihood = np.sum(CurrLogLikelihood[np.isinf(CurrLogLikelihood) == 0])

    return -LogLikelihood


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
    for rep in range(1, NrOfReplicates):
        new_k, Ks = multdirichletVect.expand_k(Counts[rep * tracks_per_rep:(rep + 1) * tracks_per_rep, :])
        k += new_k

    DBase = psi(np.sum(curr_alpha)) - psi(np.sum(k, axis=0) + np.sum(curr_alpha)) 

    for J in range(0, curr_alpha.shape[0]):
        D = psi(curr_alpha[J] + k) -  psi(curr_alpha[J])

    CurrLogLikeliehood = np.float64((D + DBase) * NrOfCounts)
    LogLikelihood[J] += np.sum(CurrLogLikeliehood[np.isinf(CurrLogLikeliehood) == 0])

    return -LogLikelihood




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
    for i in range(1, NrOfReplicates):
        RatioLikelihood += multdirichletVect.log_pdf_vect(Counts[i * tracks_per_rep:(i + 1) * tracks_per_rep, :], alpha)
    CurrLogLikelihood = RatioLikelihood * np.float64(NrOfCounts)
    LogLikelihood += np.sum(CurrLogLikelihood[np.isinf(CurrLogLikelihood) == 0])

    return -LogLikelihood



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
    for rep in range(1, NrOfReplicates):
        curr_k = Counts[rep * tracks_per_rep:(rep + 1) * tracks_per_rep, :]
        DBase -=  psi(np.sum(curr_k, axis=0) + np.sum(curr_alpha))       
    for J in range(0, curr_alpha.shape[0]):
        curr_k = Counts[J, :]
        D = psi(curr_alpha[J] + curr_k) -  psi(curr_alpha[J])
        for rep in range(1, NrOfReplicates):
            curr_k = Counts[rep * tracks_per_rep + J, :]
            D += psi(curr_alpha[J] + curr_k)  - psi(curr_alpha[J])
        CurrLogLikeliehood = np.float64((D + DBase) * np.float64(NrOfCounts))
        LogLikelihood[J] += np.sum(CurrLogLikeliehood[np.isinf(CurrLogLikeliehood) == 0])
    return -LogLikelihood


def GenerateTestSet(alpha_TC, P_I):
    '''
    This function generates a test set to test the model convergence
    '''

    NrOfSamples = 1000
    MaxExpr_TC = 5
    MaxExpr_non_TC = 10
    TestSuffStat = {}
    #Generate the samples
    NrPos = 0
    for i in range(NrOfSamples):
        #Draw wether the set is positive or negative
        I = bernoulli.rvs(P_I)
        #Draw the number of reads
        if I == 1:
            CurrNrReads = poisson.rvs(MaxExpr_TC)
            GammaSample = dirichlet(alpha_TC)
            NrPos += 1
            ReadsRep1 = multinomial(CurrNrReads, GammaSample)
            CurrNrReads = poisson.rvs(MaxExpr_TC)
            ReadsRep2 = multinomial(CurrNrReads, GammaSample)
        else:
            CurrNrReads = poisson.rvs(MaxExpr_non_TC)
            GammaSample = dirichlet(np.array([0.1,0.5,1])*1.0)
            ReadsRep1 = multinomial(CurrNrReads, GammaSample)
            CurrNrReads = poisson.rvs(MaxExpr_non_TC)
            GammaSample = dirichlet(np.array([0.1,0.5,1])*1.0)
            ReadsRep2 = multinomial(CurrNrReads, GammaSample)

        # Draw the conversions and seq errors 
        #Add the obesrvation to the suff statistics
        key = tuple(np.hstack((ReadsRep1, ReadsRep2)))
        if TestSuffStat.has_key(key):
            TestSuffStat[key] += 1
        else:
            TestSuffStat[key] = 1
    print 'Number of positives: ' + str(NrPos) + ' of ' + str(NrOfSamples)
    return TestSuffStat


def RunTest(alpha_TC, P_I):
    '''
    This function runs the paramter estimation and return a comparison
    '''

    print 'Generating test set'
    TestSuffStat = GenerateTestSet(alpha_TC, P_I)
    #estimate parameters
    print 'Estimate parameters'
    alpha_TC_est, alpha_nonTC_est, Z, P_I_est, P_NonI, IterLog = StartMLEstimationDirch(TestSuffStat, 0.2, 10)
    #Show results
    print "alpha_TC true: " + str(alpha_TC)
    print "alpha_TC est: " + str(alpha_TC_est)
    print "P_I true: " + str(P_I)
    print "P_I est: " + str(P_I_est)
    Mat =  ConvertSuffStatToMat(TestSuffStat)
    IXPos = Z[0, :] > 0.5
    IXNeg = Z[0, :] < 0.5
    plt.scatter(Mat[0, IXPos]/(Mat[0,  IXPos] + Mat[2, IXPos]), Mat[3, IXPos]/(Mat[3,  IXPos] + Mat[5, IXPos]),color = 'red')
    plt.scatter(Mat[0, IXNeg]/(Mat[0,  IXNeg] + Mat[2, IXNeg]), Mat[3, IXNeg]/(Mat[3,  IXNeg] + Mat[5, IXNeg]),color = 'blue')
    plt.show()
    return IterLog

def ConvertSuffStatToMat(SuffStat):
    '''
    This function generates from the SuffStat data structure a matrix form
    '''
    
    #Get the shape of the keys and initialize the matrix
    M = np.sum(np.array([SuffStat[key] for key in SuffStat.keys()]))
    N = len(SuffStat.keys()[0])
    Mat = np.zeros((N, M))
    Counter = 0
    #Fill the matrix
    for key in SuffStat.keys():
        Mat[:, Counter : (Counter + SuffStat[key])] = np.tile(np.array(key).T , (SuffStat[key], 1)).T
        Counter += SuffStat[key]
    
    return Mat


    
def GeneratePlots(TestMat):
    SuffStat = ModelTCConversionDirUnifSep.GenerateTestSet(np.array([2,0.1,10]), 0.01)
    Mat = ModelTCConversionDirUnifSep.ConvertSuffStatToMat(SuffStat)
    plt.scatter(Mat[0,:]/(Mat[0, :] + Mat[2,:]), Mat[3,:]/(Mat[3, :] + Mat[5,:]),color = 'red')
    SuffStat = ModelTCConversionDirUnifSep.GenerateTestSet(np.array([2,0.1,10]), 0.99)
    Mat = ModelTCConversionDirUnifSep.ConvertSuffStatToMat(SuffStat)
    plt.scatter(Mat[0,:]/(Mat[0, :] + Mat[2,:]), Mat[3,:]/(Mat[3, :] + Mat[5,:]),color = 'blue')    
    plt.show()
    
    
    SuffStat = ModelTCConversionDirUnifSep.GenerateTestSet(np.array([50,1,100]), 0.01)
    Mat = ModelTCConversionDirUnifSep.ConvertSuffStatToMat(SuffStat)
    plt.hist(Mat.T, 50)
    plt.figure()
    SuffStat = ModelTCConversionDirUnifSep.GenerateTestSet(np.array([5,1,100]), 0.99)
    Mat = ModelTCConversionDirUnifSep.ConvertSuffStatToMat(SuffStat)
    plt.hist(Mat.T, 50)
    plt.show()

    MaxNr = 50
    X = np.zeros((MaxNr, MaxNr))
    Y = np.zeros_like(X)
    
    for i in range(MaxNr):
        for j in range(MaxNr):
            alpha_TC = np.array([1,10])
            k_TC1 = np.array([i, j])
            k_TC2 = k_TC1
            X[i, j] = multdirichlet.log_pdf(k_TC1, alpha_TC) + multdirichlet.log_pdf(k_TC2, alpha_TC)
            Y[i, j] = multdirichlet.TwoBinomlog_pdf(k_TC1, k_TC2, alpha_TC)
            
    plt.imshow(X-Y, interpolation='none')
    plt.colorbar()
    plt.show()


