import os
import json

from sys import argv
from reference_store import ReferenceStore



if __name__ == '__main__':

    connection_string = argv[1] if len(argv) > 1 else 'mongodb://localhost' 
    rs = ReferenceStore(connection_string)

    references = []
    for file in os.scandir('data'):
        if file.is_file():
            with open(file.path, 'rt') as reference_file:
                references.extend( json.load(reference_file) )
    
    rs.insert(references)