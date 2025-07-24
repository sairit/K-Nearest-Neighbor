"""
@Author: Sai Yadavalli
Version: 2.1
"""

import pandas as pd
import glob
from knn import KNearestNeighbor
        

# Testing the K-Nearest-Neighbor model
model = KNearestNeighbor(3, "target")

# Combine all training and validation files
train_files = glob.glob(f"{model.path}/train/train*.csv")
validation_files = glob.glob(f"{model.path}/train/validation*.csv")

# Combine all training files into one DataFrame
train_dfs = [pd.read_csv(file) for file in train_files]
combined_train = pd.concat(train_dfs, ignore_index=True)
combined_train_file = f"{model.path}/train/combined_train.csv"
combined_train.to_csv(combined_train_file, index=False)

# Combine all validation files into one DataFrame
validation_dfs = [pd.read_csv(file) for file in validation_files]
combined_validation = pd.concat(validation_dfs, ignore_index=True)
combined_validation_file = f"{model.path}/train/combined_validation.csv"
combined_validation.to_csv(combined_validation_file, index=False)

# Call the testing method with the combined files
model.testing(1, combined_validation_file, combined_train_file, standard=True)


