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

import gffutils

# Gene ID identifiers
id_spec = {'gene': ['gene_id', 'geneID', 'ID=gene:']}


def CreateDB(FileNameGFF, FileNameDB, gene_features):
    """Create a GFF database using GFFUtils."""
    # Read the annotation file
    with open(FileNameGFF, 'r') as f:
        lines = f.readlines()

    # Keep only lines describing genes as features
    genes = '\n'.join([
        '\t'.join(line.split('\t')[:2] + ['gene'] + line.split('\t')[3:])
        for line in lines
        if (line[0] != '#' and line.split('\t')[2] in gene_features)])

    # Create the DB
    gffutils.create_db(
        data=genes, from_string=True, dbfn=FileNameDB, force=True,
        keep_order=True, merge_strategy='create_unique',
        disable_infer_transcripts=True, disable_infer_genes=True,
        sort_attribute_values=True, id_spec=id_spec)
