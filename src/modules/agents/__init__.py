REGISTRY = {}

from .rnn_agent import RNNAgent
from .transformer_agent import  TransformerAgent
REGISTRY["rnn"] = RNNAgent
REGISTRY["transformer"] = TransformerAgent
