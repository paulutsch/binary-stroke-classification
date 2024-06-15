import random
import sys

import numpy as np
import sklearn
import torch
from loguru import logger

from src.data import DataLoader, DataPreprocessor, FeatureEngineer
from src.models import LogReg, NeuralNet, RandomForest
from src.visualization import DataExplorer, ResultsExplorer

# get cli args
use_dependencies = "use-dependencies" in sys.argv
visualize_data = "skip-visualization" not in sys.argv
log_level = (
    sys.argv[sys.argv.index("--log-level") + 1] if "--log-level" in sys.argv else "INFO"
)
path_data_train = (
    sys.argv[sys.argv.index("--path-data-train") + 1]
    if "--path-data-train" in sys.argv
    else "data/raw/train.csv"
)
path_data_test = (
    sys.argv[sys.argv.index("--path-data-test") + 1]
    if "--path-data-test" in sys.argv
    else "data/raw/test.csv"
)

# set seeds
seed_value = 42
np.random.seed(seed_value)
random.seed(seed_value)
torch.manual_seed(seed_value)
sklearn.utils.check_random_state(seed_value)

# configure logging
logger.remove(0)
logger.add(sys.stderr, level=log_level.upper())

# log config vars
logger.info(f"log-level: {log_level.upper()}")
logger.info(f"visualize-data: {visualize_data}")
logger.info(f"use-dependencies: {use_dependencies}")
logger.info(f"path-data-train: {path_data_train}")
logger.info(f"path-data-test: {path_data_test}\n")

# load data
data_loader = DataLoader()

data_train_raw = data_loader.load_data(path_data_train)
X_test_raw = data_loader.load_data(path_data_test)  # no target given in test set

# visualize raw data
if visualize_data:
    data_explorer = DataExplorer()
    data_explorer.plot_distributions(data_train_raw)
    data_explorer.plot_nans(data_train_raw)
    data_explorer.print_non_numericals(data_train_raw)

# preprocess data
data_preprocessor = DataPreprocessor(data_train_raw)
data_train_preprocessed = data_preprocessor.transform_non_numericals(data_train_raw)
X_test_preprocessed = data_preprocessor.transform_non_numericals(X_test_raw)
# data_train_preprocessed = data_preprocessor.remove_nans(data_train_raw)

if visualize_data:
    data_explorer.plot_correlation_matrix(data_train_preprocessed)

# initialize models
log_reg = LogReg(use_dependencies)
# ...
