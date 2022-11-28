import os
import json

from typing import List
from pymongo import MongoClient



def extract_references(data_folder: str):
    references = []
    for file in os.scandir(data_folder):
        if file.is_file():
            with open(file.path, 'rt') as reference_file:
                references.extend( json.load(reference_file) )
    
    return references



class LandmarkData:

    def __init__(self, x: float, y: float, z: float, visibility: float):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility
    
    def as_dict(self) -> dict:
        return {
            'x': self.x,
            'y': self.y,
            'z': self.z,
            'visibility': self.visibility
        }


class ExerciseReference:

    def __init__(self, first_half: bool, exercise: str, progress: float, landmarks: List[LandmarkData]):
        self.first_half = first_half
        self.exercise = exercise
        self.progress = progress
        self.landmarks = landmarks

    def as_dict(self) -> dict:
        return {
            'first_half': self.first_half,
            'exercise': self.exercise,
            'progress': self.progress,
            'landmarks': [landmark.as_dict() for landmark in self.landmarks]
        }


class ReferenceStore:

    def __init__(self, data_folder: str=None):
        self._data = {}
        if data_folder:
            for doc in extract_references(data_folder):
                self._doc_to_data(doc)
    
    def get(self, exercise: str, first_half: bool) -> List[ExerciseReference]:
        return self._data[exercise, first_half]

    def insert(self, references: List[dict]):
        for doc in references:
            self._doc_to_data(doc)
    
    def exercises(self) -> List[str]:
        return list({ exercise for exercise, first_half in self._data })

    def _doc_to_data(self, doc: dict):
        reference = ExerciseReference(
            first_half=doc['first_half'],
            exercise=doc['exercise'],
            progress=doc['progress'],
            landmarks=[LandmarkData(**landmark) for landmark in doc['landmarks']]
        )
        self._data.setdefault((reference.exercise, reference.first_half), []).append(reference)


class ReferenceStoreMongo(ReferenceStore):

    def __init__(self, connection_string: str):
        self._connection_string = connection_string
    
        self._client = MongoClient(self._connection_string)

        self._database = self._client['smart_move']
        self._collection = self._database['exercise_references']
    
    def get(self, exercise: str, first_half: bool) -> List[ExerciseReference]:
        return [ExerciseReference(
            first_half=doc['first_half'],
            exercise=doc['exercise'],
            progress=doc['progress'],
            landmarks=[LandmarkData(**landmark) for landmark in doc['landmarks']]
        ) for doc in self._collection.find({'exercise': exercise, 'first_half': first_half})]

    def insert(self, references: List[dict]):
        self._collection.insert_many(references)
    
    def exercises(self) -> List[str]:
        return self._collection.distinct('exercise')