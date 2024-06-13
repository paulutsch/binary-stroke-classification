from data import feature_engineer
from src.data import DataLoader, DataPreprocessor
from src.models import LogReg
from src.visualization import DataExplorer, ResultsExplorer

loader = DataLoader()
loader.load_data()
loader.to_numpy()

data_explorer = DataExplorer()

model = LogReg()
model.fit()
