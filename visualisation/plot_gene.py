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


import numpy as np
import sys
import importlib
sys.path.append('../data_parsing/')
sys.path.append('../stat/')
sys.path.append('../visualisation/')
from scipy.sparse import *
import diag_event_model
import viterbi
import prettyplotlib as ppl
import brewer2mpl
import emission
import trans
import tools
import pylab as plt
from scipy.special import logsumexp
import matplotlib


matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def PlotGene(Sequences, Background, gene, IterParameters, TransitionTypeFirst = 'nonhomo', no_plot = False , Start = 0, Stop = -1, figsize=(6,8), dir_ylim=[], out_name=None):
    '''
    This function plot the coverage and the parameters for the model
    '''

    importlib.reload(diag_event_model)
    importlib.reload(emission)
    set2 = brewer2mpl.get_map('Dark2', 'qualitative', 8).mpl_colors
    TransitionParameters = IterParameters[1]
    EmissionParameters = IterParameters[0]
    TransitionType = EmissionParameters['TransitionType']
    PriorMatrix = EmissionParameters['PriorMatrix']
    NrOfStates = EmissionParameters['NrOfStates']
    
    Sequences_per_gene = PreloadSequencesForGene(Sequences, gene)
    Background_per_gene = PreloadSequencesForGene(Background, gene)

    if EmissionParameters['FilterSNPs']:
        Ix = tools.GetModelIx(Sequences_per_gene, Type='no_snps_conv', snps_thresh=EmissionParameters['SnpRatio'], snps_min_cov=EmissionParameters['SnpAbs'], Background=Background_per_gene)
    else:
        Ix = tools.GetModelIx(Sequences_per_gene)

    #2) Compute the probabilities for both states
    EmmisionProbGene = np.log(np.ones((NrOfStates, Ix.shape[0])) * (1 / np.float64(NrOfStates)))
    EmmisionProbGene_Dir = np.log(np.ones((NrOfStates, Ix.shape[0])) * (1 / np.float64(NrOfStates)))
    EmmisionProbGeneNB_fg = np.log(np.ones((NrOfStates, Ix.shape[0])) * (1 / np.float64(NrOfStates)))
    EmmisionProbGeneNB_bg = np.log(np.ones((NrOfStates, Ix.shape[0])) * (1 / np.float64(NrOfStates)))


    CurrStackSum = tools.StackData(Sequences_per_gene) 
    CurrStackVar = tools.StackData(Sequences_per_gene, add = 'no')
    nr_of_genes = len(list(Sequences.keys()))
    gene_nr_dict = {}
    for i, curr_gene in enumerate(Sequences.keys()):
        gene_nr_dict[curr_gene] = i

    #Compute the emission probapility
    for State in range(NrOfStates):
        if not EmissionParameters['ExpressionParameters'][0] == None:
            EmmisionProbGene[State, :] = emission.predict_expression_log_likelihood_for_gene(CurrStackSum, State, nr_of_genes, gene_nr_dict[gene], EmissionParameters)
            EmmisionProbGeneNB_fg[State, :] = emission.predict_expression_log_likelihood_for_gene(CurrStackSum, State, nr_of_genes, gene_nr_dict[gene], EmissionParameters)            
            if EmissionParameters['BckType'] == 'Coverage':
                EmmisionProbGene[State, :] += emission.predict_expression_log_likelihood_for_gene(tools.StackData(Background, gene, add = 'only_cov')+0, State, nr_of_genes, gene_nr_dict[gene], EmissionParameters, curr_type='bg')
                EmmisionProbGeneNB_bg[State, :] = emission.predict_expression_log_likelihood_for_gene(tools.StackData(Background, gene, add = 'only_cov')+0, State, nr_of_genes, gene_nr_dict[gene], EmissionParameters, curr_type='bg')
            if EmissionParameters['BckType'] == 'Coverage_bck':
                EmmisionProbGene[State, :] += emission.predict_expression_log_likelihood_for_gene(tools.StackData(Background, gene, add = 'only_cov')+0, State, nr_of_genes, gene_nr_dict[gene], EmissionParameters, curr_type='bg')
                EmmisionProbGeneNB_bg[State, :] = emission.predict_expression_log_likelihood_for_gene(tools.StackData(Background, gene, add = 'only_cov')+0, State, nr_of_genes, gene_nr_dict[gene], EmissionParameters, curr_type='bg')
        if not EmissionParameters['ign_diag']:
            EmmisionProbGene[State, Ix] += diag_event_model.pred_log_lik(CurrStackVar[:, Ix], State, EmissionParameters)
            EmmisionProbGene_Dir[State, Ix] = diag_event_model.pred_log_lik(CurrStackVar[:, Ix], State, EmissionParameters)
        
    #Get the transition probabilities
    if TransitionTypeFirst == 'nonhomo':
        if TransitionType == 'unif_bck' or TransitionType == 'binary_bck':
            CountsSeq = tools.StackData(Sequences_per_gene, add = 'all')
            CountsBck = tools.StackData(Background_per_gene, add = 'only_cov')
            Counts = np.vstack((CountsSeq, CountsBck))
        else:
            Counts = tools.StackData(Sequences_per_gene, add = 'all')
        TransistionProbabilities = np.float64(trans.PredictTransistions(Counts, TransitionParameters, NrOfStates, TransitionType))
    else: 
        TransistionProbabilities = np.float64(np.tile(np.log(TransitionParameters[0]), (EmmisionProbGene.shape[1],1,1)).T)       

    MostLikelyPath, LogLik = viterbi.viterbi(np.float64(EmmisionProbGene), TransistionProbabilities, np.float64(np.log(PriorMatrix)))
    for j in range(NrOfStates):
        print(str(np.sum(MostLikelyPath == j)))

    if no_plot:
        return MostLikelyPath, TransistionProbabilities, EmmisionProbGene
    #pdb.set_trace()
    fig, axes = plt.subplots(nrows=9, figsize=figsize)
    fig.subplots_adjust(hspace = 1.001)

    Counts = tools.StackData(Sequences_per_gene, gene , add = 'no')
    if Stop == -1:
        Stop = Counts.shape[1]
    if Stop == -1:
        plt_rng = np.array(list(range(Start, Counts.shape[1])))
    else:
        plt_rng = np.array(list(range(Start, Stop)))

    i = 0
    color = set2[i]
    nr_of_rep_fg = len(list(Sequences[gene]['Coverage'].keys()))
    i+=1
    Ix = repl_track_nr([2, 16], 22, nr_of_rep_fg) 
    ppl.plot(axes[0], plt_rng, (np.sum(Counts[Ix,:], axis=0))[Start:Stop], label='TC', linewidth=2, color = color)
    color = set2[i]
    i += 1
    Ix = repl_track_nr([0,1,3,5,6,7,8,10,11,12,13,15,17,18], 22, nr_of_rep_fg) 
    ppl.plot(axes[0], plt_rng, (np.sum(Counts[Ix,:], axis=0))[Start:Stop], label='NonTC', linewidth=2, color = color)
    color = set2[i]
    i += 1
    Ix = repl_track_nr([20], 22, nr_of_rep_fg) 
    ppl.plot(axes[0], plt_rng, (np.sum(Counts[Ix,:], axis=0))[Start:Stop], label='Read-ends', linewidth=2, color = color)
    color = set2[i]
    i += 1
    Ix = repl_track_nr([4,9,14,19], 22, nr_of_rep_fg) 
    ppl.plot(axes[0], plt_rng, (np.sum(Counts[Ix,:], axis=0))[Start:Stop], label='Deletions', linewidth=2, color = color)
    color = set2[i]
    i += 1
    Ix = repl_track_nr([21], 22, nr_of_rep_fg) 
    ppl.plot(axes[0], plt_rng, (np.sum(Counts[Ix,:], axis=0))[Start:Stop], label='Coverage', linewidth=2, color = color)
    color = set2[i]
    i += 1
    axes[0].set_ylabel('Counts')
    axes[0].set_xlabel('Position')
    axes[0].set_title('Coverage and Conversions')
    axes[0].get_xaxis().get_major_formatter().set_useOffset(False)

    BckCov = Background_per_gene['Coverage'][0]
    for i in range(1,len(list(Background_per_gene['Coverage'].keys()))):
        BckCov += Background_per_gene['Coverage'][str(i)]
    
    ppl.plot(axes[0], plt_rng, (BckCov.T)[Start:Stop], ls = '-', label='Bck', linewidth=2, color = color)
    ppl.legend(axes[0])

    for j in range(NrOfStates):
        color = set2[j]
        ppl.plot(axes[1], plt_rng, (TransistionProbabilities[j,j,:])[Start:Stop], label='Transition ' + str(j) + ' ' + str(j), linewidth=2, color = color)
    
    ppl.legend(axes[1])
    axes[1].set_ylabel('log-transition probability')
    axes[1].set_xlabel('Position')
    axes[1].set_title('Transition probability')
    axes[1].get_xaxis().get_major_formatter().set_useOffset(False)

    for j in range(NrOfStates):
        color = set2[j]
        ppl.plot(axes[2], plt_rng, (EmmisionProbGene[j,:][Start:Stop]), label='Emission ' + str(j) , linewidth=2, color = color)
    if EmissionParameters['BckType'] == 'Coverage_bck':
        axes[2].set_ylim((np.min(np.min(EmmisionProbGene[0:2, :][:, Start:Stop])), 1))

    ppl.legend(axes[2])
    axes[2].set_ylabel('log-GLM probability')
    axes[2].set_xlabel('Position')
    axes[2].set_title('Emission probability')
    axes[2].get_xaxis().get_major_formatter().set_useOffset(False)

    ppl.plot(axes[3], plt_rng, MostLikelyPath[Start:Stop])
    axes[3].set_ylabel('State')
    axes[3].set_xlabel('Position')
    axes[3].set_title('Most likely path')
    axes[3].get_xaxis().get_major_formatter().set_useOffset(False)

    for j in range(NrOfStates):
        color = set2[j]
        ppl.plot(axes[4], plt_rng, EmmisionProbGene_Dir[j, :][Start:Stop], label='Dir State ' + str(j) , linewidth=2, color = color)
    if len(dir_ylim) > 0:
        axes[4].set_ylim(dir_ylim)    
    ppl.legend(axes[4])
    axes[4].set_ylabel('log-DMM probability')
    axes[4].set_xlabel('Position')
    axes[4].set_title('DMM probability')
    axes[4].get_xaxis().get_major_formatter().set_useOffset(False)

    for j in range(NrOfStates):
        color = set2[j]
        ppl.plot(axes[5], plt_rng,  EmmisionProbGeneNB_fg[j, :][Start:Stop], label='NB fg ' + str(j) , linewidth=2, color = color)
    if EmissionParameters['BckType'] == 'Coverage_bck':
        axes[5].set_ylim([np.min(np.min(EmmisionProbGeneNB_fg[0:2, :][:, Start:Stop])), 1])

    ppl.legend(axes[5])
    axes[5].set_ylabel('prob')
    axes[5].set_xlabel('Position')
    axes[5].set_title('prob-fg')
    axes[5].get_xaxis().get_major_formatter().set_useOffset(False)

    for j in range(NrOfStates):
        color = set2[j]
        ppl.plot(axes[6], plt_rng, EmmisionProbGeneNB_bg[j, :][Start:Stop], label='NB bg ' + str(j) , linewidth=2, color = color)
    if EmissionParameters['BckType'] == 'Coverage_bck':
        axes[6].set_ylim([np.min(np.min(EmmisionProbGeneNB_bg[0:3, :][:, Start:Stop])), 1])
    ppl.legend(axes[6])
    axes[6].set_ylabel('prob')
    axes[6].set_xlabel('Position')
    axes[6].set_title('prob-bg')
    axes[6].get_xaxis().get_major_formatter().set_useOffset(False)

    fg_state, bg_state = emission.get_fg_and_bck_state(EmissionParameters, final_pred=True)
    ix_bg = list(range(EmmisionProbGene.shape[0]))
    ix_bg.remove(fg_state)
    FGScore = EmmisionProbGene[fg_state, :]
    AltScore = EmmisionProbGene[ix_bg,:]
    norm = logsumexp(AltScore, axis = 0)
    
    ix_ok = np.isinf(norm) + np.isnan(norm) 
    if np.sum(ix_ok) < norm.shape[0]:
        SiteScore = FGScore[ix_ok == 0] - norm[ix_ok == 0]
    else:
        print('Score problematic')
        SiteScore = FGScore
    ppl.plot(axes[7], plt_rng, SiteScore[Start:Stop])
    axes[7].set_ylabel('log-odd score')
    axes[7].set_xlabel('Position')
    axes[7].set_title('log-odd score')
    axes[7].get_xaxis().get_major_formatter().set_useOffset(False)



    FGScore = EmmisionProbGene_Dir[fg_state, :]
    AltScore = EmmisionProbGene_Dir[ix_bg,:]
    norm = logsumexp(AltScore, axis = 0)
    ix_ok = np.isinf(norm) + np.isnan(norm) 
    if np.sum(ix_ok) < norm.shape[0]:
        SiteScore = FGScore[ix_ok == 0] - norm[ix_ok == 0]
    else:
        print('Score problematic')
        SiteScore = FGScore
    ppl.plot(axes[8], plt_rng, SiteScore[Start:Stop])
    axes[8].set_ylabel('DMM log-odd score')
    axes[8].set_xlabel('Position')
    axes[8].set_title('DMM log-odd score')
    axes[8].get_xaxis().get_major_formatter().set_useOffset(False)
    if not (out_name is None):
        print('Saving result')
        fig.savefig(out_name)
    
    plt.show()

    return MostLikelyPath, TransistionProbabilities, EmmisionProbGeneNB_fg


def PlotTransistions():
    xx = np.array(list(range(0,100)))
    yy = np.array(list(range(0,100))).T
    xx, yy = np.meshgrid(xx, yy)
    Xfull = np.c_[xx.ravel(), yy.ravel()].T
    CovMat = IOHMM.GenerateFeatures(np.array(list(range(Xfull.shape[1] - 1))), Xfull)
    TempProb = TransitionParameters[1].predict_log_proba(CovMat.T)

    plt.figure(figsize=(3 * 2, 3 * 2))
    plt.subplots_adjust(bottom=.2, top=.95)

    for k in range(TempProb.shape[1]):
        plt.subplot(3, 3, 1 + k )
        imshow_handle = plt.imshow(np.hstack((np.zeros((1)),TempProb[:, k])).reshape((100, 100)), extent=(3, 9, 1, 5), origin='lower')

    ax = plt.axes([0.15, 0.04, 0.7, 0.05])
    plt.title("Probability")
    plt.colorbar(imshow_handle, cax=ax, orientation='horizontal')
    plt.show()

def repl_track_nr(ex_list, offset, nr_of_rep):
    '''                                                                                                                                                                                                                                                                                   
    This function computes for a list of tracks in one replicate additionaly the list for the second replicate                                                                                                                                                                            
    '''
    new_list = ex_list + list(np.array(ex_list) + offset)
    
    for i in range(2, nr_of_rep):
        new_list += list(np.array(ex_list) + offset * i)
    return new_list



def load_files():
    '''
    NOT TESTED
    This function loads the necessary files for standalaone plotting.
    '''
    args = parser.parse_args()
    print(args)

    #Check parameters
    if len(args.fg_libs) == 0:
        raise sys.exit('No CLIP-libraries given')

    if len(args.bg_libs) == 0:
        bg_type = 'None'
    else:
        bg_type = args.bg_type


    if args.out_dir == None:
        out_path = os.getcwd()
    else:
        out_path = args.out_dir

    # process the parameters
    if not (bg_type == 'Coverage' or  bg_type == 'Coverage_bck'):
        print('Bg-type: ' + bg_type + ' has not been implemented yet')
        return 



    #Load the gene annotation
    print('Loading gene annotation')
    GeneAnnotation = gffutils.FeatureDB(args.gene_anno_file, keep_order=True)
    GenomeDir = args.genome_dir

    #Load the reads
    print('Loading reads')
    DataOutFile = os.path.join(out_path, 'fg_reads.dat')
    Sequences = LoadReads.load_data(args.fg_libs, GenomeDir, GeneAnnotation, DataOutFile, load_from_file = (not args.overwrite_fg), save_results = True, Collapse = args.fg_collapsed)
    
    DataOutFile = os.path.join(out_path, 'bg_reads.dat')
    Background = LoadReads.load_data(args.bg_libs, GenomeDir, GeneAnnotation, DataOutFile, load_from_file = (not args.overwrite_bg), save_results = True, Collapse = args.bg_collapsed, OnlyCoverage = True)

    #Initializing parameters
    print('Initialising the parameters')
    if bg_type == 'Coverage_bck':
        NrOfStates = 4
    else:
        NrOfStates = 3

    TransMat = np.ones((NrOfStates, NrOfStates)) + np.eye(NrOfStates)
    TransMat = TransMat / np.sum(np.sum(TransMat))

    NrOfReplicates = len(args.fg_libs)
    gene = list(Sequences.keys())[0]
    alphashape = (Sequences[gene]['Variants'][0].shape[0] + Sequences[gene]['Coverage'][0].shape[0] + Sequences[gene]['Read-ends'][0].shape[0]) * NrOfStates
    EmissionParameters={}
    EmissionParameters['PriorMatrix'] = np.ones((NrOfStates, 1)) / float(NrOfStates)
    EmissionParameters['Alpha'] = np.ones((alphashape))
    EmissionParameters['Alpha'] = np.random.uniform(0.5, 1.5, size=(alphashape))
    EmissionParameters['NrOfStates'] = NrOfStates
    EmissionParameters['NrOfReplicates'] = NrOfReplicates
    EmissionParameters['LibrarySize'] = tools.estimate_library_size(Sequences)
    EmissionParameters['ExpressionParameters'] = [None, None]
    EmissionParameters['BckType'] = bg_type
    EmissionParameters['NrOfBckReplicates'] = len(args.bg_libs)
    EmissionParameters['BckLibrarySize'] =  tools.estimate_library_size(Background)
    EmissionParameters['TransitionType'] = args.tr_type
    EmissionParameters['Verbosity'] = args.verbosity
    EmissionParameters['Subsample'] = args.subs
