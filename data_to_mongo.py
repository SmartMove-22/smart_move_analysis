from sys import argv
from .reference_store import ReferenceStoreMongo, extract_references



'''
Simple script to insert the data from the 'data' folder (or another one) into a MongoDB instance
'''

if __name__ == '__main__':

    connection_string = argv[1] if len(argv) > 1 else 'mongodb://localhost' 
    data_folder = argv[2] if len(argv) > 2 else 'data'
    rs = ReferenceStoreMongo(connection_string)
    
    rs.insert(extract_references(data_folder))