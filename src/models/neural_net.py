from .base_model import BaseModel


class NeuralNet(BaseModel):
    def __init__(self, use_dependencies: bool = False):
        super(NeuralNet, self).__init__(use_dependencies, model_name="NeuralNet")
