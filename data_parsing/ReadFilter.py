import os
import sys

def FilterRead(Read, Filter):
    '''
    This function determines whether a read should be filtered or not
    '''
    if True:
        return True
    
    #Filter for length
    if Read.alen < Filter['Length']:
        return False
    #Filter for quality
    #if Read.
    return True
