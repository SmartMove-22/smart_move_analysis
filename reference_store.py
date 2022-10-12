from typing import List
from pymongo import MongoClient



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

    def __init__(self, first_half: bool, exercise: str, progress: float, landmarks: List[dict]):
        self.first_half = first_half
        self.exercise = exercise
        self.progress = progress
        self.landmarks = [LandmarkData(**landmark) for landmark in landmarks]

    def as_dict(self) -> dict:
        return {
            'first_half': self.first_half,
            'exercise': self.exercise,
            'progress': self.progress,
            'landmarks': [landmark.as_dict() for landmark in self.landmarks]
        }


class ReferenceStore:

    def __init__(self, connection_string: str):
        self._connection_string = connection_string
    
        self._client = MongoClient(self._connection_string)

        self._database = self._client['smart_move']
        self._collection = self._database['exercise_references']
    
    def get(self, exercise: str, first_half: bool) -> List[ExerciseReference]:
        return map(lambda kwargs: ExerciseReference(**kwargs), self._collection.find({'exercise': exercise, 'first_half': first_half}))

    def insert(self, references: dict):
        self._collection.insert_many(references)