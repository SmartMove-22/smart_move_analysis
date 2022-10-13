import os
import json

from sys import argv
from reference_store import ReferenceStore



'''
Simple script to insert the data from the 'data' folder (or another one) into a MongoDB instance
'''

if __name__ == '__main__':

    connection_string = argv[1] if len(argv) > 1 else 'mongodb://localhost' 
    data_folder = argv[2] if len(argv) > 2 else 'data'
    rs = ReferenceStore(connection_string)

    references = []
    for file in os.scandir(data_folder):
        if file.is_file():
            with open(file.path, 'rt') as reference_file:
                references.extend( json.load(reference_file) )
    
    rs.insert(references)