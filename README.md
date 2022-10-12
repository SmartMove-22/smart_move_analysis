# smart_move_analysis

Python module for data analysis at the SmartMove's backend.

## Scripts

- `stream_explore_knn.py`: experiment the KNNRegressor model by intermittently training and testing
- `stream_data_creator.py`: collect training data and store it in the `data` folder
- `data_explore_knn.py`: use the contents of the `data` folder to train and test the KNNRegressor model
- `data_to_mongo`: insert the contents of the `data` folder into a MongoDB instance