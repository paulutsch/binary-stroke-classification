from src.data import DataLoader, DataPreprocessor, FeatureEngineer
from src.models import LogReg, NeuralNet, RandomForest
from src.visualization import DataExplorer, ResultsExplorer

loader = DataLoader()
loader.load_data()
loader.to_numpy()

data_explorer = DataExplorer()

log_reg = LogReg()
