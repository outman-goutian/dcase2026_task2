# modeling_eat.py

from transformers import PreTrainedModel
from .configuration_eat import EATConfig
from .eat_model import EAT

class EATModel(PreTrainedModel):
    config_class = EATConfig

    def __init__(self, config: EATConfig):
        super().__init__(config)
        self.model = EAT(config)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def extract_features(self, x):
        return self.model.extract_features(x)
