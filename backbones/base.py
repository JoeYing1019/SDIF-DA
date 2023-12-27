import torch
import logging
from torch import nn
from .__init__ import methods_map

__all__ = ['ModelManager']

class MIA(nn.Module):

    def __init__(self, args):

        super(MIA, self).__init__()

        fusion_method = methods_map[args.method]
        self.model = fusion_method(args)

    def forward(self, text_feats, video_feats, audio_feats):

        video_feats, audio_feats = video_feats.float(), audio_feats.float()
        mm_model = self.model(text_feats, video_feats, audio_feats)

        return mm_model
    
    def pre_train(self, text_feats):
        mm_model = self.model.pre_train(text_feats)
        return mm_model



        
class ModelManager:

    def __init__(self, args):
        
        self.logger = logging.getLogger(args.logger_name)
        self.device = args.device
        self.model = self._set_model(args)

    def _set_model(self, args):

        model = MIA(args) 
        model.to(self.device)
        return model