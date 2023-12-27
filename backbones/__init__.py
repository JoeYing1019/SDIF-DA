from .SubNets.FeatureNets import BERTEncoder
from .FusionNets.SDIF import SDIF

text_backbones_map = {
                    'bert-base-uncased': BERTEncoder
                }

methods_map = {
    'sdif': SDIF
}