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
import re
import warnings


def GetRawCoverageFromRegion(
        SamReader, Chr, Start, Stop, Collapse=False, CovType='coverage',
        Genome='', legacy=True, mask_flank_variants=3, max_mm=2,
        rev_strand=None, ign_out_rds=False, gene_strand=0):
    """Extract coverage from a BAM-file.

    This function gets from a BAM-file the coverage and returns it as a sparse
    vector for each chromosome and strand.
    """
    # Initiate nucleotide lookup
    NuclDict = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'D': 4}

    # Prepare regular expression
    r = re.compile('([\\^]*[ACGTN]+)[0]*')

    # Compute Length of the region
    RegionLength = Stop - Start

    # Initialise the length vectors
    ret_arrays = dict()
    ret_arrays['variants'] = np.zeros((5, RegionLength), dtype=np.int32)
    ret_arrays['read-ends'] = np.zeros((2, RegionLength), dtype=np.int32)
    ret_arrays['coverage'] = np.zeros((1, RegionLength), dtype=np.int32)

    # Modify gene_strand if the reads are coming from the reverse strand
    if rev_strand is not None:
        if rev_strand == 0:
            gene_strand *= -1  # Swap the strand

    # Iterate over Reads
    iter = SamReader.fetch(Chr, Start, Stop)
    for CurrRead in iter:
        # Check for mismatches
        if CurrRead.get_tag('NM') > max_mm:
            continue

        # Check for position whithin gene boundaries
        CurrReadstart = CurrRead.pos - Start
        if ign_out_rds:
            LastPos = (CurrReadstart
                       + sum([e[1] for e in CurrRead.cigar if e[0] != 4])
                       - 1)

            if (CurrReadstart < 0) or (LastPos > Stop - Start):
                continue

        # Check whether the read strand matches else skip this read
        if rev_strand is not None:
            # Check if read is paired
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

        # Compute relative positions of read
        CurrReadstart = CurrRead.pos - Start
        FirstPos = CurrReadstart
        LastPos = FirstPos + sum([e[1] for e in CurrRead.cigar if e[0] != 4])
        # Check ho many reads the current read represents (Collapsing)
        if Collapse:
            Mult = CurrRead.qname.split('-')
            if len(Mult) == 1:
                raise Exception('Error: Collapsing of read: ' + CurrRead.qname)
            Mult = int(Mult[-1])
        else:
            Mult = 1

        # Processing for variants
        if 'variants' in CovType:
            GlobalVariantPos = GetVariantsFromRead(CurrRead, r)
            # Transform letters into numbers, A - 0, C - 1, G - 2, T - 3
            for e in GlobalVariantPos:
                if e[1] == 'N':
                    continue
                # Check if variant falls into flanks
                if (e[0] - Start) < (FirstPos + mask_flank_variants):
                    continue
                elif (e[0] - Start) > (LastPos - 1 - mask_flank_variants):
                    continue
                # Check whether the variant lies outside of the gene
                if e[0] - Start < 0 or e[0] - Start >= RegionLength:
                    continue
                # Process the variant
                ret_arrays['variants'][NuclDict[e[1]], e[0] - Start] += Mult

        # Processing for read extremities
        if 'read-ends' in CovType:
            if CurrRead.flag & 1:
                # Check if the read is paired end.
                # If yes chose the outer ends as ends.
                if FirstPos >= 0:
                    if ((CurrRead.flag & 128) > 0) & ((CurrRead.flag & 32) > 0):
                        ret_arrays['read-ends'][0, CurrReadstart] += Mult
                    if ((CurrRead.flag & 64) > 0) & ((CurrRead.flag & 32) > 0):
                        ret_arrays['read-ends'][1, CurrReadstart] += Mult
                if LastPos < RegionLength:
                    if ((CurrRead.flag & 16) > 0) & ((CurrRead.flag & 64) > 0):
                        ret_arrays['read-ends'][1, LastPos - 1] += Mult
                    if ((CurrRead.flag & 16) > 0) & ((CurrRead.flag & 128) > 0):
                        ret_arrays['read-ends'][0, LastPos - 1] += Mult
            else:
                if FirstPos >= 0:
                    if ((CurrRead.flag & 16) > 0):
                        ret_arrays['read-ends'][1, CurrReadstart] += Mult
                    else:
                        ret_arrays['read-ends'][0, CurrReadstart] += Mult
                if LastPos < RegionLength:
                    if ((CurrRead.flag & 16) > 0):
                        ret_arrays['read-ends'][0, LastPos - 1] += Mult
                    else:
                        ret_arrays['read-ends'][1, LastPos] += Mult

        # Processing for coverage
        for cig in CurrRead.cigar:
            # Check which type to get the coverage for
            if cig[0] == 0:
                _start = max(0, CurrReadstart)
                _end = min(RegionLength, max(0, CurrReadstart + cig[1]))
                ret_arrays['coverage'][0, _start:_end] += Mult
            if cig[0] != 4:
                CurrReadstart += cig[1]

    return ret_arrays


def GetVariantsFromRead(CurrRead, r):
    """Parse variants from read.

    Takes a pySAM read and returns variants based on the MD Tag the Variants
    and their absolute positions.
    """
    # Get the sequence
    Seq = CurrRead.seq

    # Get the MD tag
    Tag = CurrRead.get_tag('MD')

    # Split the string
    SplitTag = [e for e in r.split(Tag) if len(e) > 0]
    if len(SplitTag) == 1:
        return []

    # Convert the groups to a list of local positions and nucleotides
    TempPos = 0
    Pos = []
    for i in range(0, len(SplitTag)):
        if SplitTag[i].isdigit():
            # Increase the counter by the number of positions where there is no
            # mismatch
            TempPos += int(SplitTag[i])
        else:
            if SplitTag[i][0] == '^':  # Check if it is a deletion
                continue
            else:
                for l in range(len(SplitTag[i])):
                    Pos.append([TempPos, Seq[TempPos]])
                    TempPos += 1

    # Convert the local positions from Pos to global positions.
    CurrGlobalPos = CurrRead.pos
    GlobalPos = []

    # Iterate over the segments of the cigar string
    for cig in CurrRead.cigar:
        # Split the positions in Pos into the ones faling into the current
        # CIGAR segement and the rest

        # Sequence match
        if cig[0] == 0:
            CurrSeg = [e for e in Pos if e[0] < cig[1]]
            Pos = [[e[0] - cig[1], e[1]] for e in Pos if e[0] >= cig[1]]
            for e in CurrSeg:
                GlobalPos.append([CurrGlobalPos + e[0], e[1]])
            CurrGlobalPos += cig[1]

        # Insertion to the reference
        elif cig[0] == 1:
            continue

        # Deletion from the reference
        elif cig[0] == 2:
            for temp_pos in range(cig[1]):
                GlobalPos.append([CurrGlobalPos + temp_pos, 'D'])
            CurrGlobalPos += cig[1]
            continue

        # Skipped region from the reference
        elif cig[0] == 3:
            CurrGlobalPos += cig[1]

        # Soft clipping (clipped sequences present in SEQ)
        elif cig[0] == 4:
            continue

        # Hard clipping (clipped sequences NOT present in SEQ)
        elif cig[0] == 5:
            continue
            CurrGlobalPos += cig[1]

        # Padding (silent deletion from padded reference)
        elif cig[0] == 6:
            continue

        # Sequence match
        elif cig[0] == 7:
            CurrSeg = [e for e in Pos if e[0] < cig[1]]
            Pos = [[e[0] - cig[1], e[1]] for e in Pos if e[0] >= cig[1]]
            for e in CurrSeg:
                GlobalPos.append([CurrGlobalPos + e[0], e[1]])
            CurrGlobalPos += cig[1]

        # Sequence mismatch
        elif cig[0] == 8:
            CurrSeg = [e for e in Pos if e[0] < cig[1]]
            Pos = [[e[0] - cig[1], e[1]] for e in Pos if e[0] >= cig[1]]
            for e in CurrSeg:
                GlobalPos.append([CurrGlobalPos + e[0], e[1]])
            CurrGlobalPos += cig[1]

        else:
            warnings.warn("Encountered unhandled CIGAR character in read "
                          + CurrRead.qname)
            CurrGlobalPos += cig[1]

    return GlobalPos
