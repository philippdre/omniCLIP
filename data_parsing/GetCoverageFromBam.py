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


import os
import sys
import pysam
import h5py
import numpy as np
import re


def GetRawCoverageFromBam(InFile, HDF5OutFile, Collapse = False, CovType = 'coverage', Genome = ''):
    '''
    This function gets from a BAM-file the coverage and returns it as a sparse vector for each chromosome and strand
    '''
    
    #Initialize the HDF5 outfile
    OutFile = h5py.File(HDF5OutFile, 'w')
    
    #Open the Bam-file 
    SamReader = pysam.Samfile(InFile,'rb')
    print 'Parsing: ' + InFile
    
    #Get chromsome lengths from Header
    ChromosomeLengths = {}
    for Entry in SamReader.header['SQ']:
        ChromosomeLengths[Entry['SN']] = Entry['LN']
        
    #Iterate over chromosomes
    for Chr in ChromosomeLengths.keys():
        print 'Processing chromsome: ' + Chr
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


def GetRawCoverageFromRegion(SamReader, Chr, Start, Stop, Collapse = False, CovType = 'coverage', Genome = '', legacy = True, mask_flank_variants=3, max_mm=2):
    '''
    This function gets from a BAM-file the coverage and returns it as a sparse vector for each chromosome and strand
    '''

    #Initate nucleotide lookup
    NuclDict = {'A':0, 'C':1, 'G':2, 'T':3, 'D':4}
    
    #Compute Length of the region
    RegionLength = Stop - Start 

    #Initialise the length vectors
    if CovType == 'variants':#Check type to get the coverage for
        CurrChrCoverage = np.zeros((5, RegionLength), dtype = np.int32)
    else:
        CurrChrCoverage = np.zeros((1, RegionLength), dtype = np.int32)

    #iterate over Reads
    iter = SamReader.fetch(Chr, Start, Stop)
    for CurrRead in iter:
        if CurrRead.get_tag('NM') > max_mm:
            continue

        CurrReadstart = CurrRead.pos - Start 
        if CovType == 'variants':
            FirstPos = CurrReadstart
            LastPos = FirstPos + np.sum(np.array([e[1] for e in CurrRead.cigar if e[0] != 4])) - 1 
            GlobalVariantPos = GetVariantsFromRead(CurrRead)

            #Transform letters into numbers, A - 0, C - 1, G - 2, T - 3
            for e in GlobalVariantPos:
                #Check if variant falls into flanks
                if (e[0] - Start < FirstPos + mask_flank_variants) or (e[0] - Start > LastPos - mask_flank_variants):
                    #pdb.set_trace()
                    continue
                #Check whether the variant lies outside of the gene
                if e[0] - Start < 0 or e[0] - Start >= CurrChrCoverage.shape[1]:
                    continue
                if e[1] == 'N':
                    continue
                #Process the variant
                if Collapse:
                    Mult = CurrRead.qname.split('-')
                    if len(Mult) == 1:
                        raise Exception('Error: Collapsing of read: ' + CurrRead.qname)
                    CurrChrCoverage[NuclDict[e[1]], e[0] - Start] += int(Mult[-1])
                else:
                    CurrChrCoverage[NuclDict[e[1]], e[0] - Start] += 1

        elif CovType == 'read-ends':
            if CurrRead.flag & 1:
                #Check if the read is paried end. If yes chose the outer ends as ends
                FirstPos = CurrReadstart
                LastPos = FirstPos + np.sum(np.array([e[1] for e in CurrRead.cigar if e[0] != 4]))
                if Collapse:
                    Mult = CurrRead.qname.split('-')
                    if len(Mult) == 1:
                        raise Exception('Error: Collapsing of read: ' + CurrRead.qname)
                    if FirstPos >= 0:
                        if ((CurrRead.flag & 128) > 0) & ((CurrRead.flag & 32) > 0):
                            CurrChrCoverage[0, CurrReadstart] += int(Mult[-1])
                        if ((CurrRead.flag & 64) > 0) & ((CurrRead.flag & 32) > 0):
                            CurrChrCoverage[0, CurrReadstart] += int(Mult[-1])
                    if LastPos < CurrChrCoverage.shape[1]:
                        if ((CurrRead.flag & 16) > 0) & ((CurrRead.flag & 64) > 0):
                            CurrChrCoverage[0, LastPos - 1] += int(Mult[-1])
                        if ((CurrRead.flag & 16) > 0) & ((CurrRead.flag & 128) > 0):
                            CurrChrCoverage[0, LastPos - 1] += int(Mult[-1])
                else:
                    if FirstPos >= 0:
                        if ((CurrRead.flag & 128) > 0) & ((CurrRead.flag & 32) > 0):
                            CurrChrCoverage[0, CurrReadstart] += 1
                        if ((CurrRead.flag & 64) > 0) & ((CurrRead.flag & 32) > 0):
                            CurrChrCoverage[0, CurrReadstart] += 1
                    if LastPos < CurrChrCoverage.shape[1]:
                        if ((CurrRead.flag & 16) > 0) & ((CurrRead.flag & 64) > 0):
                            CurrChrCoverage[0, LastPos - 1] += 1
                        if ((CurrRead.flag & 16) > 0) & ((CurrRead.flag & 128) > 0):
                            CurrChrCoverage[0, LastPos - 1] += 1
            else:
                FirstPos = CurrReadstart
                LastPos = FirstPos + np.sum(np.array([e[1] for e in CurrRead.cigar if e[0] != 4]))  
                if Collapse:
                    Mult = CurrRead.qname.split('-')
                    if len(Mult) == 1:
                        raise Exception('Error: Collapsing of read: ' + CurrRead.qname)
                    if FirstPos >= 0:
                        CurrChrCoverage[0, CurrReadstart] += int(Mult[-1])
                    if LastPos < CurrChrCoverage.shape[1]:
                        CurrChrCoverage[0, LastPos - 1] += int(Mult[-1])
                else:
                    if FirstPos >= 0:
                        CurrChrCoverage[0, CurrReadstart] += 1
                    if LastPos < CurrChrCoverage.shape[1]:
                        CurrChrCoverage[0, LastPos - 1] += 1
        else:
            for Entry in CurrRead.cigar:
               #Check which type to get the coverage for
                if CovType == 'coverage':
                    if Entry[0] == 0:
                        if Collapse:
                            Mult = CurrRead.qname.split('-')
                            if len(Mult) == 1:
                                raise Exception('Error: Collapsing of read: ' + CurrRead.qname)
                            CurrChrCoverage[0, max(0, CurrReadstart):min(CurrChrCoverage.shape[1], max(0, CurrReadstart + Entry[1]))] += int(Mult[-1])
                        else:
                            CurrChrCoverage[0, max(0, CurrReadstart):min(CurrChrCoverage.shape[1], max(0, CurrReadstart + Entry[1]))] += 1
                else:
                    raise Exception('Error: unkown agument for covtype: ' + CovType)
                if Entry[0] !=4:
                    CurrReadstart += Entry[1]

    if legacy and CovType == 'variants':
        CurrChrCoverage = CurrChrCoverage[0:4,:]

    return CurrChrCoverage


def GetCoverageFromBam(InFile, HDF5OutFile, Collapse = False):
    '''
    This function gets from a BAM-file the coverage and returns it as a sparse vector for each chromosome and strand 
    '''
    GetRawCoverageFromBam(InFile, HDF5OutFile, Collapse, 'coverage')


def GetDeletionsFromBAM(InFile, HDF5OutFile, Collapse = False):
    '''
    This function  gets from a BAM-file the deletions and returns them as a sparse vector for each chromosome  and strand
    '''
    GetRawCoverageFromBam(InFile, HDF5OutFile, Collapse, 'deletions')
    
   
def GetReadEndsFromBAM(InFile, HDF5OutFile, Collapse = False):
    '''
    This function  gets from a BAM-file the deletions and returns them as a sparse vector for each chromosome  and strand
    '''
    GetRawCoverageFromBam(InFile, HDF5OutFile, Collapse, 'read_ends')
     
    

def GetVariantsFromBAM(InFile, HDF5OutFile, Collapse = False):
    '''
    This function  gets from a BAM-file the Variants and returns them as a sparse vector for each chromosome
    Variants are coded by a four bit flag where the first two bytes code for the reference and the last two for the observed variants and strand
    '''
    GetRawCoverageFromBam(InFile, HDF5OutFile, Collapse, 'variants')
    

def GetVariantsFromRead(CurrRead):
    '''
    This function takes a pysam read and returns based on the MD Tag the Variants and their absolute positions
    '''
    # Get the sequence
    Seq = CurrRead.seq
    Tag = ''
     
    # Get the MD tag
    TagList = [e[1] for e in CurrRead.tags if e[0] == 'MD']
    if len(TagList) == 0:
        warnings.warn("Warning: No MD tag found in read " + CurrRead.qname)
        return []
    else:
        Tag = TagList[0]
        
    # Split the string
    SplitTag = [e for e in re.split('([A-Z]+|\^[A-Z]+)',Tag) if len(e)>0]
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
                #TempPos += len(SplitTag[i]) - 1
            else:
                for l in range(len(SplitTag[i])):
                    PosList.append([TempPos, Seq[TempPos]])
                    #PosList.append([TempPos, SplitTag[i][l]])
                    TempPos += 1
        
    # Convert the local positions from PosList to global positions.
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
    
