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


from Bio import SeqIO
from collections import defaultdict
from scipy.sparse import csr_matrix
import GetCoverageFromBam
import gzip
import h5py
import numpy as np
import os
import pysam


def get_data_handle(file_path, write=False):
    """Return the h5py file handle."""
    if write:
        return h5py.File(file_path, 'r+')
    else:
        return h5py.File(file_path, 'r')


def close_data_handles(handle=False, handles=False):
    """Close all opened H5PY file handles."""
    # tables.file._open_files.close_all()
    if handle:
        handle.close()
    if handles:
        for _handle in handles:
            _handle.close()


def load_data(bam_files, genome_dir, gene_annotation, out_file, Collapse=False,
              OnlyCoverage=False, mask_flank_variants=3, max_mm=2,
              ign_out_rds=False, rev_strand=None):
    """Read the data from the bam-files."""
    SeqFile = h5py.File(out_file, 'w')

    if OnlyCoverage:
        print('Loading coverage only')

    # Defining vars
    trdict = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}

    print("Parsing the gene annotation")
    Genes = [gene for gene in gene_annotation.features_of_type('gene')
             if '_PAR_Y' not in gene.id]

    # Create a dictionary of genes by chromosome
    genes_chr_dict = defaultdict(list)
    for gene in Genes:
        genes_chr_dict[gene.chrom].append(gene)

    # Create a list of cromosomes
    Chrs = [chrom for chrom in list(genes_chr_dict.keys())
            if chrom.lower() != 'chrm']

    # 3. Iterate over the CurrChromosomes
    for CurrChr in Chrs:
        print('Processing ' + CurrChr)
        # Get the genome
        CurrChrFile = os.path.join(genome_dir, CurrChr + '.fa.gz')
        if not os.path.isfile(CurrChrFile):
            print('Warning, chromosome not found: ' + CurrChrFile)
            continue
        with gzip.open(CurrChrFile, 'rt') as handle:
            record_dict = SeqIO.to_dict(SeqIO.parse(handle, 'fasta'))
            CurrChrSeq = record_dict[CurrChr]

        for i, bam_file in enumerate(bam_files):
            # Iterate over the Genes
            SamReader = pysam.Samfile(bam_file, 'rb')
            for gene in genes_chr_dict[CurrChr]:
                start = gene.start
                stop = gene.stop
                gene_id = gene.id.split('.')[0]
                strand = 1 if gene.strand == '+' else -1
                # get the data
                CovType = ['coverage']
                if not OnlyCoverage:
                    CovType += ['variants', 'read-ends']

                ret_arrays = GetCoverageFromBam.GetRawCoverageFromRegion(
                    SamReader, CurrChr, start, stop, Collapse=Collapse,
                    CovType=CovType, Genome='', legacy=False,
                    mask_flank_variants=mask_flank_variants,
                    max_mm=max_mm, ign_out_rds=ign_out_rds,
                    gene_strand=strand, rev_strand=rev_strand)

                Coverage = ret_arrays['coverage']
                if gene_id not in SeqFile:
                    SeqFile.create_group(gene_id)
                if 'SummedCoverage' not in SeqFile[gene_id]:
                    SeqFile[gene_id].create_group('SummedCoverage')

                # Save the total coverage
                SeqFile[gene_id]['SummedCoverage'].create_dataset(
                    str(i), data=Coverage, compression="gzip",
                    compression_opts=9, chunks=Coverage.shape)

                if not OnlyCoverage:
                    Variants = ret_arrays['variants']
                    ReadEnds = ret_arrays['read-ends']

                    Coverage = (Coverage
                                - np.sum(Variants, axis=0)
                                - np.sum(ReadEnds, axis=0))

                    # Get the TC conversions
                    CurrSeq = str(CurrChrSeq.seq[start:stop]).upper()
                    GeneSeq = np.zeros((1, len(CurrSeq)), dtype=np.uint8)
                    GeneSeq[0, :] = np.array(
                        [trdict[e] for e in list(CurrSeq)])

                    # Check that group already exists, otherwise create it
                    if gene_id not in SeqFile:
                        SeqFile.create_group(gene_id)

                    if 'Coverage' not in SeqFile[gene_id]:
                        SeqFile[gene_id].create_group('Variants')
                        SeqFile[gene_id].create_group('Coverage')
                        SeqFile[gene_id].create_group('Read-ends')

                    # Ignore Variants at N positions
                    Variants[:, GeneSeq[0, :] == 4] = 0
                    Variants[Variants < 0] = 0

                    non_zer_var = np.where(Variants)
                    ij = np.vstack((
                        GeneSeq[0, non_zer_var[1]] * 5 + non_zer_var[0],
                        non_zer_var[1]))
                    Variants_sparse = csr_matrix(
                        (Variants[np.where(Variants)], ij),
                        shape=(20, len(CurrSeq)))
                    Variants = Variants_sparse.toarray()
                    del non_zer_var, ij

                    SeqFile[gene_id]['Variants'].create_group(str(i))
                    variant_fields = ['data', 'indptr', 'indices', 'shape']
                    for field in variant_fields:
                        SeqFile[gene_id]['Variants'][str(i)].create_dataset(
                            field, data=getattr(Variants_sparse, field))

                    # Only save the positions where a read in one of the
                    # replicates occured
                    SeqFile[gene_id]['Read-ends'].create_dataset(
                        str(i), data=ReadEnds, compression="gzip",
                        compression_opts=9)

                    del Variants, ReadEnds, CurrSeq, GeneSeq, Variants_sparse

                else:
                    if 'Coverage' not in SeqFile[gene_id]:
                        SeqFile[gene_id].create_group('Coverage')

                SeqFile[gene_id]['Coverage'].create_dataset(
                    str(i), data=Coverage, compression="gzip",
                    compression_opts=9)

                if str(i) == '0':
                    SeqFile[gene_id].create_dataset('strand', data=strand)
                del Coverage

        del CurrChrSeq
        del record_dict

        SeqFile.close()
