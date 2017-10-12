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
sys.path.append('../stat/')

from Bio import SeqIO
from collections import defaultdict
from scipy.misc import logsumexp
from scipy.sparse import *
from scipy.stats import nbinom
import cPickle
import diag_event_model
import emission
import evaluation
import gzip
import h5py
import itertools
import multiprocessing
import numpy as np
import os
import re
import time
import trans
import viterbi

#@profile
def GetModelIx(Sequences, Type = 'all', snps_thresh = 0.4, snps_min_cov = 10, Background = None):
    '''
    This function returns the positions for at which the emission should be Computed
    '''

    if Type == 'all':
        Ix = np.sum(StackData(Sequences, add = 'all'), axis = 0) > 0
    elif Type == 'no_snps_all':
        if not Background == None:
            Bck_var = np.sum(StackData(Background, add = 'only_var'), axis = 0)
            Bck = np.sum(StackData(Background, add = 'only_cov'), axis = 0)
            Bck_ratio = np.zeros_like(Bck_var)
            ix = Bck > 0
            Bck_ratio[ix] = np.float64(Bck_var[ix]) / np.float64(Bck[ix])
            #Positions that have a minimla coverage and have a snp ratio above the trheshold are not considered
            Ix_bg = (Bck > snps_min_cov) * (Bck_ratio > snps_thresh) > 0
            Ix_fg = np.sum(StackData(Sequences, add = 'all'), axis = 0) > 0
            Ix = (Ix_fg > 0) * (Ix_bg == 0) > 0 
    elif Type == 'no_snps_conv':
        if not Background == None:
            Bck_var = np.sum(StackData(Background, add = 'only_var'), axis = 0)
            Bck = np.sum(StackData(Background, add = 'only_cov'), axis = 0)
            Bck_ratio = np.zeros_like(Bck_var)
            ix = Bck > 0
            Bck_ratio[ix] = np.float64(Bck_var[ix]) / np.float64(Bck[ix])
            #Positions that have a minimla coverage and have a snp ratio above the trheshold are not considered
            Ix_bg = (Bck > snps_min_cov) * (Bck_ratio > snps_thresh) > 0
            Ix_fg = np.sum(StackData(Sequences, add = 'nocov'), axis = 0) > 0
            Ix = (Ix_fg > 0) * (Ix_bg == 0) > 0
    elif Type == 'Conv':
        Ix = np.sum(StackData(Sequences, add = 'nocov'), axis = 0) > 0

    else:
        Ix = np.sum(StackData(Sequences, add = 'nocov'), axis = 0) > 0
    return Ix

#@profile
def StackData(Sequences, add = 'all', use_strand = 'True'):
    '''
    This function stacks the data for a gene
    '''

    Track_strand_map = [18, 17, 16, 15, 19, 13, 12, 11, 10, 14, 8, 7, 6, 5, 9, 3, 2, 1, 0, 4]

    nr_of_repl = len(Sequences['Coverage'].keys())
    gene_len = Sequences['Coverage']['0'].shape[1]
    if (add == 'all') or (add == 'nocov'):
        CurrStack = np.zeros((nr_of_repl, gene_len))        
        for rep in Sequences['Coverage'].keys():
            CurrStack[int(rep), :] += Sequences['Variants'][rep].sum(axis=0)
            CurrStack[int(rep), :] += Sequences['Read-ends'][rep].sum(axis=0)
            CurrStack[int(rep), :] += Sequences['Coverage'][rep].sum(axis=0)

    elif add == 'only_cov':
        #Check if the cariants are substracted from the coverage:
        if 'Variants' in Sequences:  
            CurrStack = np.zeros((nr_of_repl, gene_len))        
            for rep in Sequences['Coverage'].keys():
                CurrStack[int(rep), :] += Sequences['Variants'][rep].sum(axis = 0)
                CurrStack[int(rep), :] += Sequences['Read-ends'][rep].sum(axis=0)
                CurrStack[int(rep), :] += Sequences['Coverage'][rep].sum(axis=0)                
        else:
            CurrStack = np.zeros((nr_of_repl, gene_len))        
            for rep in Sequences['Coverage'].keys():
                CurrStack[int(rep), :] += Sequences['Coverage'][rep].sum(axis=0)
    elif add == 'only_var':
        CurrStack = np.zeros((nr_of_repl, gene_len))        
        for rep in Sequences['Coverage'].keys():
            CurrStack[int(rep), :] += Sequences['Variants'][rep].sum(axis = 0)

    else:
        CurrArr = []
        for rep in Sequences['Variants'].keys():
            if use_strand:
                if Sequences['strand'] == 1:
                    CurrArr.append(np.vstack((Sequences['Variants'][rep], Sequences['Read-ends'][rep], Sequences['Coverage'][rep] )))
                else:
                    CurrArr.append(np.vstack((Sequences['Variants'][rep][Track_strand_map, :], Sequences['Read-ends'][rep], Sequences['Coverage'][rep] )))
            else:
                CurrArr.append(np.vstack((Sequences['Variants'][rep], Sequences['Read-ends'][rep], Sequences['Coverage'][rep] )))
        CurrStack = np.array(np.vstack((CurrArr)))
        del CurrArr

    CurrStack[CurrStack < 0] = 0
    return CurrStack

#@profile
def PreloadSequencesForGene(Sequences, gene):
    '''
    This function stacks the data for a gene
    '''

    Sequences_per_gene = {}
    for key in Sequences[gene]:
        if isinstance(Sequences[gene][key], h5py.Dataset):
            Sequences_per_gene[key] = Sequences[gene][key].value
        else:
            Sequences_per_gene[key] = {}
            for rep in Sequences[gene][key]:
                Sequences_per_gene[key][rep] = Sequences[gene][key][rep].value

    return Sequences_per_gene


#@profile
def GetSuffStat(Sequences, Background, Paths, NrOfStates, Type, ResetNotUsedStates = True, EmissionParameters=None):
    '''
    This function computes for each CurrPath state a set of suffcient statistics:
    '''

    #Initialize the sufficent statistcs variable
    print "Getting suffcient statistic"
    t = time.time()
    SuffStat = {}
    for CurrState in range(NrOfStates):
        SuffStat[CurrState] = defaultdict(int)        #SuffStat[CurrState] = Collections.Counter()

    try:
        Sequences.close()
    except:
        pass
    try:
        Background.close()
    except:
        pass     
    Sequences = h5py.File(EmissionParameters['DataOutFile_seq'], 'r')
    Background = h5py.File(EmissionParameters['DataOutFile_bck'], 'r')

    #Fil the sufficent statistcs variable
    for gene in Sequences.keys():
        rep = Sequences[gene]['Coverage'].keys()[0]
        CurrGenePath = Paths[gene]

        #Stack the matrizes together and convert to dense matrix
        Sequences_per_gene = PreloadSequencesForGene(Sequences, gene)
        Background_per_gene = PreloadSequencesForGene(Background, gene)
        if Type == 'Conv':
            CurrStack = StackData(Sequences_per_gene, add = 'variants')
        else:
            CurrStack = StackData(Sequences_per_gene, add = 'all')

        if EmissionParameters['FilterSNPs']:
            if Type == 'Conv':
                Ix = GetModelIx(Sequences_per_gene, Type='no_snps_conv', snps_thresh=EmissionParameters['SnpRatio'], snps_min_cov=EmissionParameters['SnpAbs'], Background=Background_per_gene)
            else:
                Ix = GetModelIx(SSequences_per_gene, Type)
        else:
            Ix = GetModelIx(Sequences_per_gene, Type)

        NonZero = np.sum(CurrStack, axis = 0) > 0
        
        #Determine the nonzeros elements
        for CurrState in xrange(NrOfStates):
            if EmissionParameters['mask_ovrlp']:
                CurrIx = Ix * (Sequences_per_gene['mask'][rep][0, :] == 0) * NonZero * (CurrGenePath == CurrState) > 0
            else:
                CurrIx = Ix * NonZero * (CurrGenePath == CurrState) > 0
            
            data = CurrStack[:,CurrIx].T
            ncols = data.shape[1]
            dtype = data.T.dtype.descr * ncols
            struct = data.view(dtype)

            vals, val_counts = np.unique(struct, return_counts=True)

            #Save the tuples and how many times they have been seen so far.
            for curr_val, curr_count in itertools.izip(vals, val_counts): 
                SuffStat[CurrState][tuple(curr_val)] += curr_count
            
            #Treat the 0 tuple seperately for speed improvment
            if len(Ix) == 0:
                continue
            
            NullIx = (NonZero == 0) * (CurrGenePath == CurrState) > 0
            if np.sum(NullIx) == 0:
                continue
            NullCount = np.sum(NullIx)
            if NullCount > 0:
                NullTuple = np.zeros_like(CurrStack[:, 0])
                NullTuple = tuple(NullTuple.T)
                SuffStat[CurrState][NullTuple] += NullCount
        
        del CurrStack, NonZero, CurrGenePath, Ix

    print 'Done: Elapsed time: ' + str(time.time() - t)

    return SuffStat

#@profile
def GetSuffStatBck(Sequences, Background, Paths, NrOfStates, Type, ResetNotUsedStates = True, EmissionParameters=None):
    '''
    This function computes for each CurrPath state a set of suffcient statistics:
    '''

    #Initialize the sufficent statistcs variable
    print "Getting suffcient statistic"
    t = time.time()
    SuffStatBck = {}

    fg_state, bg_state = emission.get_fg_and_bck_state(EmissionParameters, final_pred=True)

    SuffStatBck[fg_state] = defaultdict(int)

    try:
        Sequences.close()
    except:
        pass
    try:
        Background.close()
    except:
        pass     
    Sequences = h5py.File(EmissionParameters['DataOutFile_seq'], 'r')
    Background = h5py.File(EmissionParameters['DataOutFile_bck'], 'r')

    #Fil the sufficent statistcs variable
    for gene in Sequences.keys():
        rep = Background[gene]['Coverage'].keys()[0]
        CurrGenePath = Paths[gene]

        #Stack the matrizes together and convert to dense matrix
        Background_per_gene = PreloadSequencesForGene(Background, gene)
        Sequences_per_gene = PreloadSequencesForGene(Sequences, gene)
        if Type == 'Conv':
            CurrStack = StackData(Background_per_gene, add = 'variants')
        else:
            CurrStack = StackData(Background_per_gene, add = 'all')

        if EmissionParameters['FilterSNPs']:
            if Type == 'Conv':
                Ix = GetModelIx(Background_per_gene, Type='no_snps_conv', snps_thresh=EmissionParameters['SnpRatio'], snps_min_cov=EmissionParameters['SnpAbs'], Background=Background_per_gene)
            else:
                Ix = GetModelIx(Background_per_gene, Type)
        else:
            Ix = GetModelIx(Background_per_gene, Type)

        NonZero = np.sum(CurrStack, axis = 0) > 0

        #Determine the nonzeros elements
        CurrState = fg_state
        
        CurrIx = Ix * NonZero > 0
        if EmissionParameters['mask_ovrlp']:
            CurrIx = Ix * (Sequences_per_gene['mask'][rep][0, :] == 0) * NonZero * (CurrGenePath == CurrState) > 0
        else:
            CurrIx = Ix * NonZero * (CurrGenePath == CurrState) > 0

        data = CurrStack[:,CurrIx].T
        ncols = data.shape[1]
        dtype = data.T.dtype.descr * ncols
        struct = data.view(dtype)

        vals, val_counts = np.unique(struct, return_counts=True)

        #Save the tuples and how many times they have been seen so far.
        for curr_val, curr_count in itertools.izip(vals, val_counts): 
            SuffStatBck[CurrState][tuple(curr_val)] += curr_count

        #Treat the 0 tuple seperately for speed improvment
        if len(Ix) == 0:
            continue
        NullIx = (NonZero == 0) * (CurrGenePath == CurrState) > 0
        if np.sum(NullIx) == 0:
            continue
        NullCount = np.sum(NullIx)
        if NullCount > 0:
            NullTuple = np.zeros_like(CurrStack[:, 0])
            NullTuple = tuple(NullTuple.T)
            SuffStatBck[CurrState][NullTuple] += NullCount
        
        del CurrStack, NonZero, CurrGenePath, Ix
    
    print 'Done: Elapsed time: ' + str(time.time() - t)

    return SuffStatBck

#@profile
def ConvertSuffStatToArrays(SuffStat):
    '''
    This function converts the Suffcient statistics into a list of arrays 
    '''

    #Initializse the return values
    Counts = {}
    NrOfCounts = {}
    for State in SuffStat.keys():
        if len(SuffStat[State].keys()) == 0:
            print "empyt suffcient keys "
            Counts[State] = np.array([])
            NrOfCounts[State] = np.array([])
        else:
            Counts[State] = np.array([np.array(key) for key in SuffStat[State].keys()]).T
            NrOfCounts[State] = np.tile(np.array(SuffStat[State].values()), (1,1))
    
    return Counts, NrOfCounts

#@profile
def GetSuffStatisticsForGeneReplicate(HDF5File, GeneAnnotation, TrackInfo, GenomeDir, OutFile):
    '''
    This function iterates over the HDF5File to get the sufficient statistics
    TrackInfo contains the information on the tracks that shouel be uses
    '''

    print "computing SuffStat for list of genes"
    f = h5py.File(HDF5File, 'r')

    #Initialize return dict
    GeneConversionEvents = {}
    
    #Get the Genes
    print "Parsing the gene annotation"
    Iter = GeneAnnotation.features_of_type('gene')
    Genes = []
    for gene in Iter:
        Genes.append(gene)
    #3. Get the data
    # Get the cromosomes
    Chrs = []
    for gene in Genes:
        Chrs.append(gene.chrom)
    Chrs = list(set(Chrs))
    #3. Iterate over the CurrChromosomes 
    for CurrChr in Chrs:
        #Make sure that the CurrChr is indeed a CurrChromosome and not header information
        print 'Processing ' + CurrChr
        #Get the genome
        CurrChrFile  = os.path.join(GenomeDir, CurrChr + '.fa.gz')
        handle = gzip.open(CurrChrFile, "rU")
        record_dict = SeqIO.to_dict(SeqIO.parse(handle, "fasta"))
        CurrChrSeq = record_dict[CurrChr]
        handle.close()
      
        #Iterate over the Genes
        for gene in Genes:
            if gene.chrom == CurrChr:
                Start = gene.start
                Stop = gene.stop 
                Strand =  1 if gene.strand == '+' else -1
                GeneLength = Stop - Start

                #Get all conversions
                ConvNr = np.zeros((len(TrackInfo['ReplicatesCov']), GeneLength), dtype=np.int)
                CovNr = np.zeros_like(ConvNr, dtype=np.int)
                ConvNrC = np.zeros_like(ConvNr, dtype=np.int)
                ConvNrG = np.zeros_like(ConvNr, dtype=np.int)
                
                for i, Repl in enumerate(TrackInfo['ReplicatesCovA']):
                    ConvNr[i, :] += np.int32(f[CurrChr][Repl, Start : Stop])
                for i, Repl in enumerate(TrackInfo['ReplicatesCovC']):
                    ConvNr[i, :] += np.int32(f[CurrChr][Repl, Start : Stop])
                    ConvNrC[i, :] += np.int32(f[CurrChr][Repl, Start : Stop])
                for i, Repl in enumerate(TrackInfo['ReplicatesCovG']):
                    ConvNr[i, :] += np.int32(f[CurrChr][Repl, Start : Stop])
                    ConvNrG[i, :] += np.int32(f[CurrChr][Repl, Start : Stop])
                for i, Repl in enumerate(TrackInfo['ReplicatesCovT']):
                    ConvNr[i, :] += np.int32(f[CurrChr][Repl, Start : Stop])
                #Get the Coverage at the conversions location
                for i, Repl in enumerate(TrackInfo['ReplicatesCov']):
                    CovNr[i, :] += np.int32(f[CurrChr][Repl, Start : Stop])
                #Determine the number of replicates
                NrOfReplicates = i

                #Get the TC conversions
                CurrSeq = str(CurrChrSeq.seq[Start : Stop])
                GenomeIsA = np.zeros((1, len(CurrSeq)))
                GenomeIsT = np.zeros((1, len(CurrSeq)))
                Ix = [m.start() for m in re.finditer('A', CurrSeq.upper())]
                GenomeIsA[0, Ix] = 1
                del Ix
                Ix = [m.start() for m in re.finditer('T', CurrSeq.upper())]
                GenomeIsT[0, Ix] = 1
                del Ix
                
                ConvNrTC = np.zeros((len(TrackInfo['ReplicatesCov']), GeneLength))
                if (GenomeIsA.shape[1] != ConvNrG.shape[1]) or (GenomeIsT.shape[1] != ConvNrC.shape[1]):
                    print 'Warning: Size mismatch at ' + CurrChr
                    continue
                if Strand == -1:
                    ConvNrTC += (np.tile(GenomeIsA, (NrOfReplicates, 1)) * ConvNrG)
                    PossibleConversion = GenomeIsA
                else:
                    ConvNrTC += (np.tile(GenomeIsT, (NrOfReplicates, 1)) * ConvNrC)
                    PossibleConversion = GenomeIsT
                ConvNrNonTC = ConvNr - ConvNrTC
                #Define the return value for gene
                CurrGene = {}
                CurrGene['ConvNrTC'] = csr_matrix(ConvNrTC)
                CurrGene['ConvNrNonTC'] = csr_matrix(ConvNrNonTC)
                #Only save the positions where a read in one of the replicates occured
                CurrGene['PossibleConversion'] = csr_matrix(PossibleConversion)
                
                CurrGene['CovNr'] = csr_matrix(CovNr - ConvNrTC - ConvNrNonTC)
                GeneConversionEvents[gene.id.split('.')[0]] = CurrGene
                del CurrGene, ConvNrTC, ConvNrNonTC, PossibleConversion
                del GenomeIsA, GenomeIsT, CurrSeq, CovNr, ConvNr, ConvNrC, ConvNrG
        del CurrChrSeq
        del record_dict
    del Genes
    with gzip.GzipFile(OutFile, 'w') as f:
            cPickle.dump(GeneConversionEvents, f)
    return 


def repl_track_nr(ex_list, offset):
    '''
    This function computes for a list of tracks in one replicate additionaly the list for the second replicate
    '''

    new_list = ex_list + list(np.array(ex_list) + offset)

    return new_list

#@profile
def GeneratePred(Paths, Sequences, Background, IterParameters, GeneAnnotation, OutFile, fg_state = 1, noise_state = 0, seq_file='', bck_file='', pv_cutoff=0.05):
    '''
    This function writes the predictions
    '''

    TransitionParameters = IterParameters[1]
    EmissionParameters = IterParameters[0]

    merge_neighbouring_sites = False
    minimal_site_length = 1

    #Predict the sites
    print 'Score peaks'
    try:
        Sequences.close()
    except:
        pass
    try:
        Background.close()
    except:
        pass     
    Sequences = h5py.File(EmissionParameters['DataOutFile_seq'], 'r')
    Background = h5py.File(EmissionParameters['DataOutFile_bck'], 'r')

    ScoredSites = GetSites(Paths, Sequences, Background, EmissionParameters, TransitionParameters, 'nonhomo', fg_state, merge_neighbouring_sites, minimal_site_length, seq_file=seq_file, bck_file=bck_file)
    
    Sequences = h5py.File(seq_file, 'r')
    Background = h5py.File(bck_file, 'r')
    
    print 'Write peaks'
    #Write the results
    WriteResults(Sequences, Background, ScoredSites, OutFile, GeneAnnotation)

    generate_bed(OutFile, pv_cutoff=0.05)
    return


def generate_bed(file, pv_cutoff=0.05):
    df = pd.read_table(file)

    #Determine the cutoff for the p-vales aften Bonferroni
    cutoff = np.log( 0.05 /  df.shape[0])
    df = df[df['pv'] < cutoff]

    
    #Make the names unique
    df['Gene'] = df['Gene'] + ['_' + str(i) for i in range(df.shape[0])]

    #Transform the Score to be betweeen 0 and 1000
    df['SiteScore'] /= np.max(df['SiteScore']) * 1000


    #Compute thickstart and end
    df['ThickStart'] = 0
    df['ThickStop'] = 0

    df.ix[df['Strand'] == 1, 'ThickStart'] = df['Start'] + df['max_pos']
    df.ix[df['Strand'] == -1, 'ThickStart'] = df['Stop'] - df['max_pos'] 


    df.ix[df['Strand'] == 1, 'ThickStop'] = df['Start'] + df['max_pos'] + 1
    df.ix[df['Strand'] == -1, 'ThickStop'] = df['Stop'] - df['max_pos'] + 1 

    #rename the strands
    df.ix[df['Strand'] == 1, 'Strand'] = '+'
    df.ix[df['Strand'] == -1, 'Strand'] = '-'

    #Keep only the columns that are relevant for the bed-file
    df = df[['ChrName', 'Start', 'Stop', 'Gene', 'SiteScore', 'Strand' , 'ThickStart', 'ThickStop']].sort_values('SiteScore', ascending=False)

    #Write the output
    df.to_csv(file.replace('.txt', '.bed') , sep = '\t', header=False, index=False)
    return

#@profile
def GetSites(Paths, Sequences, Background, EmissionParameters, TransitionParameters, TransitionTypeFirst, fg_state, merge_neighbouring_sites, minimal_site_length, seq_file='', bck_file=''):
    ''' 
    This function get the predicted sites
    '''
    NrOfStates = EmissionParameters['NrOfStates']
    TransitionType = EmissionParameters['TransitionType']
    alpha = EmissionParameters['Diag_event_params']
    
    #Extract the paths
    Sites = evaluation.convert_paths_to_sites(Paths, fg_state, merge_neighbouring_sites, minimal_site_length)

    np_proc = EmissionParameters['NbProc']
    nr_of_genes = len(Sequences.keys())
    gene_nr_dict = {}
    for i, curr_gene in enumerate(Sequences.keys()):
        gene_nr_dict[curr_gene] = i

    sites_keys = [key for key in Sites.keys() if len(Sites[key]) > 0]
    f = lambda key, Sites=Sites, nr_of_genes=nr_of_genes, gene_nr_dict=gene_nr_dict, seq_file=seq_file, bck_file=bck_file, EmissionParameters=EmissionParameters, TransitionParameters=TransitionParameters, TransitionTypeFirst=TransitionTypeFirst, fg_state=fg_state, merge_neighbouring_sites=merge_neighbouring_sites, minimal_site_length=minimal_site_length: (Sites[key], key, nr_of_genes, gene_nr_dict[key], seq_file, bck_file, EmissionParameters, TransitionParameters, TransitionTypeFirst, fg_state, merge_neighbouring_sites, minimal_site_length)

    data = itertools.imap(f, sites_keys)
    
    number_of_processes = np_proc
    
    Sequences.close()
    Background.close()
    if np_proc == 1:
        ScoredSites = dict([GetSitesForGene(curr_slice) for curr_slice in data])
    else:    
        pool = multiprocessing.Pool(number_of_processes, maxtasksperchild=10)
        results = pool.imap(GetSitesForGene, data, chunksize=1)
        pool.close()
        pool.join()        
        ScoredSites = dict([res for res in results])
    for key in ScoredSites.keys():
        if len(ScoredSites[key]) == 0:
            del ScoredSites[key]

    
    return ScoredSites

#@profile
def GetSitesForGene(data):
    '''
    This function determines for each gene the score of the sites
    '''

    #Computing the probabilities for the current gene

    Sites, gene, nr_of_genes, gene_nr, seq_file, bck_file, EmissionParameters, TransitionParameters, TransitionTypeFirst, fg_state, merge_neighbouring_sites, minimal_site_length = data
    #Turn the Sequence and Bacground objects into dictionaries again such that the subsequent methods for using these do not need to be modified
    if len(Sites) == 0:
        return gene, []

    NrOfStates = EmissionParameters['NrOfStates']
    TransitionType = EmissionParameters['TransitionType']
    
    Sites = dict([(gene, Sites)])

    Sequences = h5py.File(EmissionParameters['DataOutFile_seq'], 'r')
    Background = h5py.File(EmissionParameters['DataOutFile_bck'], 'r')

    Sequences_per_gene = PreloadSequencesForGene(Sequences, gene)
    Background_per_gene = PreloadSequencesForGene(Background, gene)

    Ix = GetModelIx(Sequences_per_gene, Type='all')

    if np.sum(Ix) == 0:
        return gene, []

    if EmissionParameters['FilterSNPs']:
        Ix = GetModelIx(Sequences_per_gene, Type='no_snps_conv', snps_thresh=EmissionParameters['SnpRatio'], snps_min_cov=EmissionParameters['SnpAbs'], Background=Background_per_gene)
    else:
        Ix = GetModelIx(Sequences_per_gene, Type='Conv')

    #Only compute the emission probability for regions where a site is
    ix_sites = np.zeros_like(Ix)
    ix_sites_len = Ix.shape[0]
    for currsite in Sites[gene]:
        ix_sites[max(0, currsite[0] - 1) : min(ix_sites_len, currsite[1] + 1)] = 1
    ix_sites = ix_sites == 1

    #2) Compute the probabilities for both states
    EmmisionProbGene = np.log(np.ones((NrOfStates, Ix.shape[0])) * (1 / np.float64(NrOfStates)))
    CurrStackSum = StackData(Sequences_per_gene)
    CurrStackVar = StackData(Sequences_per_gene, add = 'no')
    CurrStackSumBck = StackData(Background_per_gene, add = 'only_cov')
    EmmisionProbGeneDir = np.zeros_like(EmmisionProbGene)

    if EmissionParameters['glm_weight'] < 0.0:
        weight1 = 1.0
        weight2 = 1.0
    elif EmissionParameters['glm_weight'] == 0.0:
        weight1 = 0.0000001
        weight2 = 1.0 - weight1 
    elif EmissionParameters['glm_weight'] == 1.0:
        weight1 = 0.9999999
        weight2 = 1.0 - weight1 
    else:
        weight1 = EmissionParameters['glm_weight'] 
        weight2 = (1.0 - EmissionParameters['glm_weight']) 

        
    for State in range(NrOfStates):
        EmmisionProbGene[State, ix_sites] = np.log(weight1) + emission.predict_expression_log_likelihood_for_gene(CurrStackSum[:, ix_sites], State, nr_of_genes, gene_nr, EmissionParameters)
        if EmissionParameters['BckType'] == 'Coverage':
            EmmisionProbGene[State, ix_sites] += np.log(weight1) + emission.predict_expression_log_likelihood_for_gene(CurrStackSumBck[:, ix_sites], State, nr_of_genes, gene_nr, EmissionParameters, 'bg')
        if EmissionParameters['BckType'] == 'Coverage_bck':
            EmmisionProbGene[State, ix_sites] += np.log(weight1) + emission.predict_expression_log_likelihood_for_gene(CurrStackSumBck[:, ix_sites], State, nr_of_genes, gene_nr, EmissionParameters, 'bg')
        if not EmissionParameters['ign_out_rds']:
            EmmisionProbGeneDir[State, Ix] = np.log(weight2) + diag_event_model.pred_log_lik(CurrStackVar[:, Ix], State, EmissionParameters)
        EmmisionProbGene[State, Ix] += np.log(weight2) + EmmisionProbGeneDir[State, Ix]

    if TransitionType == 'unif_bck' or TransitionType == 'binary_bck':
        CountsSeq = StackData(Sequences_per_gene, add = 'all')
        CountsBck = StackData(Background_per_gene, add = 'only_cov')
        Counts = np.vstack((CountsSeq, CountsBck))
    else:
        Counts = StackData(Sequences_per_gene, add = 'all')
    

    Score = EmmisionProbGene
    CurrStack = CurrStackVar
    
    #Compute the scores when staying in the sam state
    RowIx = range(16) + range(17, 38) + range(39,44)
    strand = Sequences_per_gene['strand']

    #Get the coverages for the froeground and background
    CountsSeq = StackData(Sequences_per_gene, add = 'only_cov')
    CountsBck = StackData(Background_per_gene, add = 'only_cov')

    if strand == 0:
        strand = -1
    #Since we the transition probabilty is the same for all States we do not need to compute it for the bayes factor
    #this list contains the returned sites
    sites = []
    for currsite in Sites[gene]:
        mean_mat_fg, var_mat_fg, mean_mat_bg, var_mat_bg, counts_fg, counts_bg = ComputeStatsForSite(CountsSeq, CountsBck, currsite, fg_state, nr_of_genes, gene_nr, EmissionParameters)

        site = {}
        site['Start'] = currsite[0]
        site['Stop'] = currsite[1]
        site['Strand'] = strand
        site['SiteScore'] = EvaluateSite(Score, currsite, fg_state)
        site['Coverage'] = np.sum(np.sum(Counts[:, site['Start'] : site['Stop']], axis=0))
        site['TC'] = np.sum(np.sum(CurrStack[[16, 38], site['Start'] : site['Stop']], axis=0))
        site['NonTC'] = np.sum(np.sum(CurrStack[RowIx, site['Start'] : site['Stop']], axis=0))
        site['mean_mat_fg'] = mean_mat_fg
        site['var_mat_fg'] = var_mat_fg
        site['mean_mat_bg'] = mean_mat_bg
        site['var_mat_bg'] = var_mat_bg
        site['counts_fg'] = counts_fg
        site['counts_bg'] = counts_bg

        p = mean_mat_fg / var_mat_fg
        n = (mean_mat_fg ** 2) / (var_mat_fg - mean_mat_fg)
        site['pv'] = nbinom.logsf(counts_fg, n, p)
        site['max_pos'] = get_max_position(Score, currsite, fg_state, strand)
        site['dir_score'] = EvaluateSite(EmmisionProbGeneDir, currsite, fg_state)
        if site['SiteScore'] < 0.0:
            continue
        sites.append(site)

    Sequences.close()
    Background.close()

    return gene, sites

#@profile
def ComputeStatsForSite(CountsSeq, CountsBck, Site, fg_state, nr_of_genes, gene_nr, EmissionParameters):
    '''
    Get the score for a Site
    '''

    Start = Site[0]
    Stop = Site[1]

    mean_mat_fg, var_mat_fg = emission.get_expected_mean_and_var(CountsSeq[:, Start : (Stop + 1)], fg_state, nr_of_genes, gene_nr, EmissionParameters, curr_type = 'fg')
    mean_mat_bg, var_mat_bg = emission.get_expected_mean_and_var(CountsBck[:, Start : (Stop + 1)], fg_state, nr_of_genes, gene_nr, EmissionParameters, curr_type = 'bg')
    
    mean_mat_fg = np.sum(np.sum(mean_mat_fg, axis=0))
    var_mat_fg = np.sum(np.sum(var_mat_fg, axis=0))
    mean_mat_bg = np.sum(np.sum(mean_mat_bg, axis=0))
    var_mat_bg = np.sum(np.sum(var_mat_bg, axis=0))
    counts_fg = np.sum(np.sum(CountsSeq[:, Start : (Stop + 1)], axis = 0))
    counts_bg = np.sum(np.sum(CountsBck[:, Start : (Stop + 1)], axis = 0))


    return mean_mat_fg, var_mat_fg, mean_mat_bg, var_mat_bg, counts_fg, counts_bg


def get_max_position(Score, Site, fg_state, strand):
    '''
    Get the site where the score is maximal 
    '''

    Start = Site[0]
    Stop = Site[1]
    ix_bg = range(Score.shape[0])
    ix_bg.remove(fg_state)
    FGScore = Score[fg_state, Start : (Stop + 1)]
    AltScore = Score[ix_bg, Start : (Stop + 1)]

    norm = logsumexp(AltScore, axis = 0)
    
    ix_ok = np.isinf(norm) + np.isnan(norm) 
    if np.sum(ix_ok) < norm.shape[0]:
        SiteScore = FGScore[ix_ok == 0] - norm[ix_ok == 0]
    else:
        print 'Score problematic'
        SiteScore = FGScore

    max_pos = np.int64(np.round(np.mean(np.where(SiteScore == np.max(SiteScore))[0])))
    if strand == -1:
        pos = Stop - Start - max_pos
    else:
        pos = max_pos

    return pos


def EvaluateSite(Score, Site, fg_state):
    '''
    Get the score for a Site
    '''

    Start = Site[0]
    Stop = Site[1]
    ix_bg = range(Score.shape[0])
    ix_bg.remove(fg_state)
    FGScore = np.sum(Score[fg_state, Start : (Stop + 1)])
    AltScore = np.sum(Score[ix_bg, Start : (Stop + 1)], axis = 1)
    norm = logsumexp(AltScore)
    if not (np.isinf(norm) or np.isnan(norm)):
        SiteScore = FGScore - norm
    else:
        print 'Score problematic'
        SiteScore = FGScore

    return SiteScore

#@profile
def WriteResults(Sequences, Background, ScoredSites, OutFile, GeneAnnotation):
    '''
    This function writes the sites into a result file
    '''

    #Get the gene annotation
    Iter = GeneAnnotation.features_of_type('gene')
    Genes = []
    for gene in Iter:
        Genes.append(gene)

    #Print the results
    fid = open(OutFile, 'w')
    Header = '\t'.join(['Gene', 'ChrName', 'Start', 'Stop', 'Strand', 'SiteScore', 'Coverage', 'TC', 'NonTC', 'mean_mat_fg', 'var_mat_fg', 'mean_mat_bg', 'var_mat_bg', 'counts_fg', 'counts_bg', 'pv', 'max_pos', 'dir_score']) + '\n'
    fid.write(Header)
    for gene in Genes:
        gene_name = gene.id.split('.')[0]
        if not ScoredSites.has_key(gene_name):
            continue
        #Transform the Coordinates
        for site in ScoredSites[gene_name]:
            #Process the current site 
            CurrLine = '\t'.join([gene_name, gene.chrom, str(GetGenomicCoord(gene, site['Start'])), str(GetGenomicCoord(gene, site['Stop'])), 
                                str(site['Strand']), str(site['SiteScore']), str(site['Coverage']), str(site['TC']), str(site['NonTC']), 
                                str(site['mean_mat_fg']), str(site['var_mat_fg']), str(site['mean_mat_bg']), str(site['var_mat_bg']), 
                                str(site['counts_fg']), str(site['counts_bg']), str(site['pv']), str(site['max_pos']), str(site['dir_score'])]) + '\n'

            fid.write(CurrLine)
    fid.close()

    return


def GetGenomicCoord(gene, Coord):

    return gene.start + Coord 


def get_track_strand_order_change():
    '''
    This function determines who the order of the variant rows must be changed for genes on the negative strand
    '''

    #Create the ordering of the nucleotides
    A = []
    for n in ['a','c','g','t']:
        TT = []
        for m in ['a','c','g','t', 'd']:
            TT.append(n+m)
        A.append(TT)

    #Convert to matrix
    A = np.array(A)    
    B = A[ [3,2,1,0],:][:, [3,2,1,0,4 ]]

    A = A.flatten().tolist()
    B = B.flatten().tolist()

    Order = [B.index(n) for n in A]

    return Order


def estimate_library_size(Sequences):
    '''
    This function estimates the library size of all samples
    '''

    lib_size_dict = defaultdict(list)
    lib_size_red = defaultdict(int)
    #Get the gene expressions
    for gene in Sequences.keys():
        for key in Sequences[gene]['Coverage'].keys():
            lib_size_dict[key].append(Sequences[gene]['Coverage'][key].value.sum())

    #Compute the (weighted) median of non zero genes 
    for key in lib_size_dict:
        exprs = np.array([g for g in lib_size_dict[key] if g > 0])
        temp_median = np.median(exprs)
        temp_med_floor = np.floor(temp_median)
        temp_med_ceil = np.ceil(temp_median)
        tot_floor = np.sum(exprs == temp_med_floor)
        tot_ceil = np.sum(exprs == temp_med_ceil)
        med =  ((temp_med_floor * tot_floor) + (temp_med_ceil * tot_ceil)) / (tot_floor + tot_ceil)
        med = temp_median
        lib_size_red[key] = med

    return lib_size_red


def GetMostLikelyPath(MostLikelyPaths, Sequences, Background, EmissionParameters, TransitionParameters, TransitionTypeFirst, RandomNoise = False):
    '''
    This function computes the most likely path. Ther are two options, 'homo' and 'nonhomo' for TransitionType.
    This specifies whether the transition probabilities should be homogenous or non-homogenous.
    '''

    MostLikelyPaths = {}
    alpha = EmissionParameters['Diag_event_params']
    PriorMatrix = EmissionParameters['PriorMatrix']
    NrOfStates = EmissionParameters['NrOfStates']
    np_proc = EmissionParameters['NbProc']
    LogLikelihood = 0

    TransitionType = EmissionParameters['TransitionType']

    #Iterate over genes
    nr_of_genes = len(Sequences.keys())
    gene_nr_dict = {}
    for i, curr_gene in enumerate(Sequences.keys()):
        gene_nr_dict[curr_gene] = i
        
    print "Computing most likely path"
    t = time.time()
    for i, gene in enumerate(Sequences.keys()):
        if i % 1000 == 0:
            sys.stdout.write('.')
            sys.stdout.flush()
        #Score the state sequences
        #1) Determine the positions where an observation is possible
        Sequences_per_gene = PreloadSequencesForGene(Sequences, gene)
        Background_per_gene = PreloadSequencesForGene(Background, gene)

        Ix = GetModelIx(Sequences_per_gene, Type='all')

        if np.sum(Ix) == 0:
            MostLikelyPaths[gene] = 2 * np.ones((0, Ix.shape[0]), dtype=np.int)
            continue 

        if EmissionParameters['FilterSNPs']:
            Ix = GetModelIx(Sequences_per_gene, Type='no_snps_conv', snps_thresh=EmissionParameters['SnpRatio'], snps_min_cov=EmissionParameters['SnpAbs'], Background=Background_per_gene)
        else:
            Ix = GetModelIx(Sequences_per_gene)

        #2) Compute the probabilities for both states
        EmmisionProbGene = np.ones((NrOfStates, Ix.shape[0])) * (1 / np.float64(NrOfStates))
        
        CurrStackSum = StackData(Sequences_per_gene)
        CurrStackVar = StackData(Sequences_per_gene, add = 'no')
        
        for State in range(NrOfStates):
            if not EmissionParameters['ExpressionParameters'][0] == None:
                #EmmisionProbGene[State, :] = FitBinoDirchEmmisionProbabilities.ComputeStateProbForGeneNB_unif(CurrStack, alpha, State, EmissionParameters)
                EmmisionProbGene[State, :] = emission.predict_expression_log_likelihood_for_gene(CurrStackSum, State, nr_of_genes, gene_nr_dict[gene], EmissionParameters)
                if EmissionParameters['BckType'] == 'Coverage':
                    EmmisionProbGene[State, :] += emission.predict_expression_log_likelihood_for_gene(StackData(Background, gene, add = 'only_cov'), State, nr_of_genes, gene_nr_dict[gene], EmissionParameters, 'bg')
                if EmissionParameters['BckType'] == 'Coverage_bck':
                    EmmisionProbGene[State, :] += emission.predict_expression_log_likelihood_for_gene(StackData(Background, gene, add = 'only_cov'), State, nr_of_genes, gene_nr_dict[gene], EmissionParameters, 'bg')

            
            EmmisionProbGene[State, Ix] += diag_event_model.pred_log_lik(CurrStackVar[:, Ix], State, EmissionParameters)

        if RandomNoise:
            EmmisionProbGene = np.logaddexp(EmmisionProbGene, np.random.uniform(np.min(EmmisionProbGene[np.isfinite(EmmisionProbGene)]) - 4, np.min(EmmisionProbGene[np.isfinite(EmmisionProbGene)]) -1, EmmisionProbGene.shape)) #Add some random noise 
            
        #Get the transition probabilities
        if TransitionTypeFirst == 'nonhomo':
            if TransitionType == 'unif_bck' or TransitionType == 'binary_bck':
                CountsSeq = StackData(Sequences_per_gene, add = 'all')
                CountsBck = StackData(Background, gene, add = 'only_cov')
                Counts = np.vstack((CountsSeq, CountsBck))
            else:
                Counts = StackData(Sequences_per_gene, add = 'all')
            TransistionProbabilities = np.float64(trans.PredictTransistions(Counts, TransitionParameters, NrOfStates, TransitionType))
        else: 
            TransistionProbabilities = np.float64(np.tile(np.log(TransitionParameters[0]), (EmmisionProbGene.shape[1],1,1)).T)

        #Perform Viterbi algorithm and append Path
        CurrPath, Currloglik = viterbi.viterbi(np.float64(EmmisionProbGene), TransistionProbabilities, np.float64(np.log(PriorMatrix)))
        MostLikelyPaths[gene] = CurrPath
        #Compute the logliklihood of the gene
        
        LogLikelihood += Currloglik
        del TransistionProbabilities, EmmisionProbGene, CurrStackSum, CurrStackVar
    
    print '\nDone: Elapsed time: ' + str(time.time() - t)
    
    return MostLikelyPaths, LogLikelihood

#@profile
def ParallelGetMostLikelyPath(MostLikelyPaths, Sequences, Background, EmissionParameters, TransitionParameters, TransitionTypeFirst, RandomNoise = False, chunksize = 1):
    '''
    This function computes the most likely path. Ther are two options, 'homo' and 'nonhomo' for TransitionType.
    This specifies whether the transition probabilities should be homogenous or non-homogenous.
    '''

    for gene in MostLikelyPaths.keys():
        del MostLikelyPaths[gene]
    np_proc = EmissionParameters['NbProc']
    nr_of_genes = len(Sequences.keys())
    gene_nr_dict = {}
    for i, curr_gene in enumerate(Sequences.keys()):
        gene_nr_dict[curr_gene] = i
    
    
    print "Computing most likely path"
    t = time.time()

    data = itertools.izip(Sequences.keys(), itertools.repeat(nr_of_genes), gene_nr_dict.values(), itertools.repeat(EmissionParameters), itertools.repeat(TransitionParameters), itertools.repeat(TransitionTypeFirst), itertools.repeat(RandomNoise))

    number_of_processes = np_proc
    
    Sequences.close()# Close the files befre forking them. 
    Background.close()

    if np_proc == 1:
        results = [ParallelGetMostLikelyPathForGene(curr_slice) for curr_slice in data]
    else:
        print "Spawning processes"
        pool = multiprocessing.Pool(number_of_processes, maxtasksperchild=5)
        results = pool.imap(ParallelGetMostLikelyPathForGene, data, chunksize)
        pool.close()
        pool.join()
        print "Collecting results"
        results = [res for res in results]
        
    MostLikelyPaths = dict(itertools.izip([result[0] for result in results], [result[1] for result in results]))
    #Compute the logliklihood of the gene
    LogLikelihood  = sum([result[2] for result in results])
    del results
    
    print '\nDone: Elapsed time: ' + str(time.time() - t)

    return MostLikelyPaths, LogLikelihood

#@profile
def ParallelGetMostLikelyPathForGene(data):
    ''' 
    This function computes the most likely path for a gene 
    '''
    
    gene, nr_of_genes, gene_nr, EmissionParameters, TransitionParameters, TransitionTypeFirst, RandomNoise = data
    
    #Turn the Sequence and Bacground objects into dictionaries again such that the subsequent methods for using these do not need to be modified
    Sequences = h5py.File(EmissionParameters['DataOutFile_seq'], 'r')
    Background = h5py.File(EmissionParameters['DataOutFile_bck'], 'r')

    #Parse the parameters
    alpha = EmissionParameters['Diag_event_params']
    PriorMatrix = EmissionParameters['PriorMatrix']
    NrOfStates = EmissionParameters['NrOfStates']

    TransitionType = EmissionParameters['TransitionType']

    fg_state, bg_state = emission.get_fg_and_bck_state(EmissionParameters, final_pred=True)
    fg_pen = EmissionParameters['fg_pen']
    #Score the state sequences
    #1) Determine the positions where an observation is possible

    Sequences_per_gene = PreloadSequencesForGene(Sequences, gene)
    Background_per_gene = PreloadSequencesForGene(Background, gene)

    Ix = GetModelIx(Sequences_per_gene, Type='all')

    if np.sum(Ix) == 0:
        CurrPath = 2 * np.ones((0, Ix.shape[0]), dtype=np.int)
        return  [gene, CurrPath, 0]

    if EmissionParameters['FilterSNPs']:
        Ix = GetModelIx(Sequences_per_gene, Type='no_snps_conv', snps_thresh=EmissionParameters['SnpRatio'], snps_min_cov=EmissionParameters['SnpAbs'], Background=Background_per_gene)
    else:
        Ix = GetModelIx(Sequences_per_gene)
    
    #2) Compute the probabilities for both states
    EmmisionProbGene = np.ones((NrOfStates, Ix.shape[0])) * (1 / np.float64(NrOfStates))
    
    CurrStackSum = StackData(Sequences_per_gene)
    CurrStackVar = StackData(Sequences_per_gene, add = 'no')
    CurrStackSumBck = StackData(Background_per_gene, add = 'only_cov')

    if EmissionParameters['glm_weight'] < 0.0:
        weight1 = 1.0
        weight2 = 1.0
    elif EmissionParameters['glm_weight'] == 0.0:
        weight1 = 0.0000001
        weight2 = 1.0 - weight1 
    elif EmissionParameters['glm_weight'] == 1.0:
        weight1 = 0.9999999
        weight2 = 1.0 - weight1 
    else:
        weight1 = EmissionParameters['glm_weight'] 
        weight2 = (1.0 - EmissionParameters['glm_weight']) 

    for State in range(NrOfStates):
        if not EmissionParameters['ign_GLM']:
            if isinstance(EmissionParameters['ExpressionParameters'][0], np.ndarray):
                #EmmisionProbGene[State, :] = FitBinoDirchEmmisionProbabilities.ComputeStateProbForGeneNB_unif(CurrStack, alpha, State, EmissionParameters)
                EmmisionProbGene[State, :] = np.log(weight1) + emission.predict_expression_log_likelihood_for_gene(CurrStackSum, State, nr_of_genes, gene_nr, EmissionParameters)
                if EmissionParameters['BckType'] == 'Coverage':
                    EmmisionProbGene[State, :] += np.log(weight1) + emission.predict_expression_log_likelihood_for_gene(CurrStackSumBck, State, nr_of_genes, gene_nr, EmissionParameters, 'bg')
                if EmissionParameters['BckType'] == 'Coverage_bck':
                    EmmisionProbGene[State, :] += np.log(weight1) + emission.predict_expression_log_likelihood_for_gene(CurrStackSumBck, State, nr_of_genes, gene_nr, EmissionParameters, 'bg')
        if not EmissionParameters['ign_diag']:
            EmmisionProbGene[State, Ix] += np.log(weight2) + diag_event_model.pred_log_lik(CurrStackVar[:, Ix], State, EmissionParameters)
        if State == fg_state:
            if EmissionParameters['LastIter']:
                EmmisionProbGene[State, :] -= fg_pen
    if RandomNoise:
        EmmisionProbGene = np.logaddexp(EmmisionProbGene, np.random.uniform(np.min(EmmisionProbGene[np.isfinite(EmmisionProbGene)]) - 4, np.min(EmmisionProbGene[np.isfinite(EmmisionProbGene)]) - 0.1, EmmisionProbGene.shape)) #Add some random noise 
        
    #Get the transition probabilities
    if TransitionTypeFirst == 'nonhomo':
        if TransitionType == 'unif_bck' or TransitionType == 'binary_bck':
            CountsSeq = StackData(Sequences_per_gene, add = 'all')
            CountsBck = StackData(Background_per_gene, add = 'only_cov')
            Counts = np.vstack((CountsSeq, CountsBck))
        else:
            Counts = StackData(Sequences_per_gene, add = 'all')
        TransistionProbabilities = np.float64(trans.PredictTransistions(Counts, TransitionParameters, NrOfStates, TransitionType))
    else: 
        TransistionProbabilities = np.float64(np.tile(np.log(TransitionParameters[0]), (EmmisionProbGene.shape[1],1,1)).T)
    
    CurrPath, Currloglik = viterbi.viterbi(np.float64(EmmisionProbGene), TransistionProbabilities, np.float64(np.log(PriorMatrix)))
    
    del TransistionProbabilities, EmmisionProbGene, CurrStackSum, CurrStackVar, Ix
    Sequences.close()
    Background.close()

    return [gene, CurrPath, Currloglik]

#@profile
def subsample_suff_stat(Counts, NrOfCounts, subsample_size=100000):
    '''
    This function  creates a subsample of the counts
    '''

    #iterate over the keys
    for key in Counts:
        #Determine the new sample
        new_counts = np.random.multinomial(min(subsample_size, np.sum(NrOfCounts[key][0,:])), NrOfCounts[key][0,:]/np.float64(np.sum(NrOfCounts[key][0,:])), size=1)
        
        #Check that not more sample than orignialy existing were present
        ix = new_counts > NrOfCounts[key]
        new_counts[ix] = NrOfCounts[key][ix]

        #if no count is there take the original sample
        if np.sum(new_counts) == 0:
            new_counts = NrOfCounts[key][0,:]

        ix_non_zero = new_counts > 0
        temp_counts = np.zeros((1, np.sum(new_counts > 0)))
        temp_counts[0, :] = new_counts[ix_non_zero]
        NrOfCounts[key] = temp_counts
        Counts[key] = Counts[key][:, ix_non_zero[0,:]]

    return Counts, NrOfCounts
