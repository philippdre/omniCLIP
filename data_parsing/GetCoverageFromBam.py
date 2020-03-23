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

import h5py
import numpy as np
import pysam
import re

##@profile
#@profile 
def GetRawCoverageFromBam(InFile, HDF5OutFile, Collapse = False, CovType = 'coverage', Genome = ''):
    '''
    This function gets from a BAM-file the coverage and returns it as a sparse vector for each chromosome and strand
    '''
    
    #Initialize the HDF5 outfile
    OutFile = h5py.File(HDF5OutFile, 'w')
    
    #Open the Bam-file 
    SamReader = pysam.Samfile(InFile,'rb')
    print('Parsing: ' + InFile)
    
    #Get chromsome lengths from Header
    ChromosomeLengths = {}
    for Entry in SamReader.header['SQ']:
        ChromosomeLengths[Entry['SN']] = Entry['LN']
        
    #Iterate over chromosomes
    for Chr in list(ChromosomeLengths.keys()):
        print('Processing chromsome: ' + Chr)
        #Initialise the length vectors
        if CovType == 'variants':#Check type to get the coverage for
            CurrChrCoverage = np.zeros((4, ChromosomeLengths[Chr]), dtype = np.int32)
        else:
            CurrChrCoverage = np.zeros((1, ChromosomeLengths[Chr]), dtype = np.int32)
        #iterate over Reads
        GetRawCoverageFromRegion(SamReader, Chr, 0, ChromosomeLengths[Chr], Collapse = False, CovType = 'coverage', Genome = '')
        
        OutFile.create_dataset(Chr, data = CurrChrCoverage, chunks=True, compression="gzip")
        
    OutFile.close()
    SamReader.close()

    return


##@profile
#@profile 
def GetRawCoverageFromRegion(SamReader, Chr, Start, Stop, Collapse = False, CovType = 'coverage', Genome = '', legacy = True, mask_flank_variants=3, max_mm=2, ign_out_rds=False, rev_strand=None, gene_strand=0):
    '''
    This function gets from a BAM-file the coverage and returns it as a sparse vector for each chromosome and strand
    '''

    #Initate nucleotide lookup
    NuclDict = {'A':0, 'C':1, 'G':2, 'T':3, 'D':4}

    #Prepare regular expression
    r = re.compile('([\\^]*[ACGTN]+)[0]*')
    
    #Compute Length of the region
    RegionLength = Stop - Start 

    #Initialise the length vectors

    CovType = ['coverage', 'variants', 'read-ends']

    CurrChrCoverage_cov = np.zeros((1, RegionLength), dtype = np.int32) # return array for coverage
    CurrChrCoverage_var = np.zeros((5, RegionLength), dtype = np.int32) # return array for variants
    CurrChrCoverage_read_end = np.zeros((2, RegionLength), dtype = np.int32) # return array for read-ends

    process_var = 'variants' in CovType
    process_read_end = 'read-ends' in CovType
    process_cov = 'coverage' in CovType
    #modificy gene_strand if the reads are coming from the reverse strand
    if rev_strand is not None:
        if rev_strand == 0:
            gene_strand *= -1 #Swap the strand

    
    #iterate over Reads
    iter = SamReader.fetch(Chr, Start, Stop)
    for CurrRead in iter:
        CurrReadstart = CurrRead.pos - Start 

        if ign_out_rds:
            LastPos = CurrReadstart + sum([e[1] for e in CurrRead.cigar if e[0] != 4]) - 1
            
            if (CurrReadstart < 0) or (LastPos > Stop - Start):
                continue
    
        #Check wether the read strand matches else skip this read
        if rev_strand is not None:
            #Check if read is paired
            if CurrRead.flag & 1:
                if gene_strand == -1:
                    # CurrRead.flag & 16 means read is on reverse strand
                    # CurrRead.flag & 32 means mate read is on reverse strand
                    # CurrRead.flag & 64 means read is first in pair
                    # CurrRead.flag & 128 means read is second in pair
                    if ((CurrRead.flag & 16) > 0) & ((CurrRead.flag & 64) > 0): 
                        continue
                    if ((CurrRead.flag & 16) == 0) & ((CurrRead.flag & 128) > 0):
                        continue
                else:
                    if ((CurrRead.flag & 16) > 0) & ((CurrRead.flag & 128) > 0):
                        continue
                    if ((CurrRead.flag & 16) == 0) & ((CurrRead.flag & 64) > 0):
                        continue
            else:
                if gene_strand == -1:
                    if ((CurrRead.flag & 16) > 0):
                        continue
                else:
                    if ((CurrRead.flag & 16) == 0):
                        continue

        if CurrRead.get_tag('NM') > max_mm:
            continue
        #Compute relative positions of read
        CurrReadstart = CurrRead.pos - Start 
        FirstPos = CurrReadstart
        cigar_str = CurrRead.cigar
        LastPos = FirstPos + sum([e[1] for e in cigar_str if e[0] != 4])
        #Check ho many reads the current read represents (Collapsing)
        if Collapse:
            Mult = CurrRead.qname.split('-')
            if len(Mult) == 1:
                raise Exception('Error: Collapsing of read: ' + CurrRead.qname)
            Mult = int(Mult[-1])
        else:
            Mult = 1

        if process_var:
            GlobalVariantPos = GetVariantsFromRead(CurrRead, r)
            #Transform letters into numbers, A - 0, C - 1, G - 2, T - 3
            for e in GlobalVariantPos:
                #Check if variant falls into flanks
                if (e[0] - Start < FirstPos + mask_flank_variants) or (e[0] - Start > LastPos - 1 - mask_flank_variants):
                    #pdb.set_trace()
                    continue
                #Check whether the variant lies outside of the gene
                if e[0] - Start < 0 or e[0] - Start >= RegionLength:
                    continue
                if e[1] == 'N':
                    continue
                #Process the variant
                CurrChrCoverage_var[NuclDict[e[1]], e[0] - Start] += Mult
        if process_read_end:
            if CurrRead.flag & 1:
                #Check if the read is paired end. If yes chose the outer ends as ends
                if FirstPos >= 0:
                    if ((CurrRead.flag & 128) > 0) & ((CurrRead.flag & 32) > 0):
                        CurrChrCoverage_read_end[0, CurrReadstart] += Mult
                    if ((CurrRead.flag & 64) > 0) & ((CurrRead.flag & 32) > 0):
                        CurrChrCoverage_read_end[1, CurrReadstart] += Mult
                if LastPos < RegionLength:
                    if ((CurrRead.flag & 16) > 0) & ((CurrRead.flag & 64) > 0):
                        CurrChrCoverage_read_end[1, LastPos - 1] += Mult
                    if ((CurrRead.flag & 16) > 0) & ((CurrRead.flag & 128) > 0):
                        CurrChrCoverage_read_end[0, LastPos - 1] += Mult
            else:
                if FirstPos >= 0:
                    if ((CurrRead.flag & 16) > 0):
                        CurrChrCoverage_read_end[1, CurrReadstart] += Mult
                    else:
                        CurrChrCoverage_read_end[0, CurrReadstart] += Mult
                if LastPos < RegionLength:
                    if ((CurrRead.flag & 16) > 0):
                        CurrChrCoverage_read_end[0, LastPos - 1] += Mult
                    else:
                        CurrChrCoverage_read_end[1, LastPos] += Mult
        if process_cov:
            for Entry in cigar_str:
               #Check which type to get the coverage for
                if Entry[0] == 0:
                    CurrChrCoverage_cov[0, max(0, CurrReadstart) : min(RegionLength, max(0, CurrReadstart + Entry[1]))] += Mult
                if Entry[0] !=4:
                    CurrReadstart += Entry[1]
    ret_arrays = {'coverage': CurrChrCoverage_cov, 'variants': CurrChrCoverage_var, 'read-ends': CurrChrCoverage_read_end}

    return ret_arrays


#@profile 
def GetCoverageFromBam(InFile, HDF5OutFile, Collapse = False):
    '''
    This function gets from a BAM-file the coverage and returns it as a sparse vector for each chromosome and strand 
    '''

    GetRawCoverageFromBam(InFile, HDF5OutFile, Collapse, 'coverage')


#@profile 
def GetDeletionsFromBAM(InFile, HDF5OutFile, Collapse = False):
    '''
    This function  gets from a BAM-file the deletions and returns them as a sparse vector for each chromosome  and strand
    '''

    GetRawCoverageFromBam(InFile, HDF5OutFile, Collapse, 'deletions')
    
   
#@profile 
def GetReadEndsFromBAM(InFile, HDF5OutFile, Collapse = False):
    '''
    This function  gets from a BAM-file the deletions and returns them as a sparse vector for each chromosome  and strand
    '''

    GetRawCoverageFromBam(InFile, HDF5OutFile, Collapse, 'read_ends')
     
    

#@profile 
def GetVariantsFromBAM(InFile, HDF5OutFile, Collapse = False):
    '''
    This function  gets from a BAM-file the Variants and returns them as a sparse vector for each chromosome
    Variants are coded by a four bit flag where the first two bytes code for the reference and the last two for the observed variants and strand
    '''

    GetRawCoverageFromBam(InFile, HDF5OutFile, Collapse, 'variants')
    
##@profile
#@profile 
def GetVariantsFromRead(CurrRead, r):
    '''
    This function takes a pysam read and returns based on the MD Tag the Variants and their absolute positions
    '''

    # Get the sequence
    Seq = CurrRead.seq
    Tag = ''
     
    # Get the MD tag
    Tag = CurrRead.get_tag('MD')
        
    # Split the string
    SplitTag = [e for e in r.split(Tag) if len(e)>0]
    if len(SplitTag) == 1:
        return [] 
    
    # Convvert the groups to a list of local positions and nuceotides
    TempPos = 0
    PosList = []
    for i in range(0,len(SplitTag)):
        if SplitTag[i].isdigit():
            TempPos += int(SplitTag[i]) # Increase Counter by the number of positions where there is no mismatch
        else:
            if SplitTag[i][0] == '^':# Check if it is a deletion
                continue
            else:
                for l in range(len(SplitTag[i])):
                    PosList.append([TempPos, Seq[TempPos]])
                    TempPos += 1
        
    # Convert the local positions from PosList to global positions.
    ReadPos = 0
    CurrGlobalPos = CurrRead.pos
    GlobalPos = []
 
    # iterate over the segements of the cigar string
    for Entry in CurrRead.cigar:
        # Split the positions in PosList into the ones faling into the current CIGAR segement and the rest
        if Entry[0] == 0:# Segequence match
            CurrSeg = [e for e in PosList if e[0] < Entry[1]]
            PosList = [ [e[0] - Entry[1], e[1]] for e in PosList if e[0] >= Entry[1]]
            for e in CurrSeg:
                GlobalPos.append([CurrGlobalPos + e[0], e[1]])
            CurrGlobalPos += Entry[1]
        elif Entry[0] == 1:# Insertion to the reference
            continue
        elif Entry[0] == 2:# Deletion from the reference
            for temp_pos in range(Entry[1]):
                GlobalPos.append([CurrGlobalPos + temp_pos, 'D'])
            CurrGlobalPos += Entry[1]
            continue
        elif Entry[0] == 3:# Skipped region from the reference
            CurrGlobalPos += Entry[1]
        elif Entry[0] == 4:# soft clipping (clipped sequences present in SEQ)
            continue
        elif Entry[0] == 5:# hard clipping (clipped sequences NOT present in SEQ)
            continue
            CurrGlobalPos += Entry[1]
        elif Entry[0] == 6:# padding (silent deletion from padded reference)
            continue
        elif Entry[0] == 7:# sequence match
            CurrSeg = [e for e in PosList if e[0] < Entry[1]]
            PosList = [ [e[0] - Entry[1], e[1]] for e in PosList if e[0] >= Entry[1]]
            for e in CurrSeg:
                GlobalPos.append([CurrGlobalPos + e[0], e[1]])
            CurrGlobalPos += Entry[1]
        elif Entry[0] == 8:# sequence mismatch
            CurrSeg = [e for e in PosList if e[0] < Entry[1]]
            PosList = [ [e[0] - Entry[1], e[1]] for e in PosList if e[0] >= Entry[1]]
            for e in CurrSeg:
                GlobalPos.append([CurrGlobalPos + e[0], e[1]])
            CurrGlobalPos += Entry[1]
        else:
            warnings.warn("Encountered unhandled CIGAR character in read " + CurrRead.qname)
            CurrGlobalPos += Entry[1]
    
    return GlobalPos
    
