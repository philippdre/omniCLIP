import gffutils
import sys

#@profile 
def CreateDB(FileNameGFF, FileNameDB):
    '''
    This function creates a GFF database
    '''

    db = gffutils.create_db(FileNameGFF, dbfn=FileNameDB, force=True, keep_order=True, merge_strategy='merge', sort_attribute_values=True)
    return

if __name__ == "__main__":
    FileNameGFF = sys.argv[1]
    
    FileNameDB = sys.argv[2]
    print('Creating annotation database')
    CreateDB(FileNameGFF, FileNameDB)


