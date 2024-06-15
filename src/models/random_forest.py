from .base_model import BaseModel


class RandomForest(BaseModel):
    def __init__(self, use_dependencies: bool = False):
        super(RandomForest, self).__init__(use_dependencies, model_name="RandomForest")
