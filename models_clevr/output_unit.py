import torch
from torch import nn

from . import ops as ops
from .config import cfg
#BRYCE CODE
import numpy as np
#BRYCE CODE

class Classifier(nn.Module):
    def __init__(self, num_choices):
        super().__init__()
        self.outQuestion = ops.Linear(cfg.CMD_DIM, cfg.CTX_DIM)
        in_dim = 3 * cfg.CTX_DIM if cfg.OUT_QUESTION_MUL else 2 * cfg.CTX_DIM
        self.classifier_layer = nn.Sequential(
            nn.Dropout(1 - cfg.outputDropout),
            ops.Linear(in_dim, cfg.OUT_CLASSIFIER_DIM),
            nn.ELU(),
            nn.Dropout(1 - cfg.outputDropout),
            ops.Linear(cfg.OUT_CLASSIFIER_DIM, num_choices))

    def forward(self, x_att, vecQuestions):
        eQ = self.outQuestion(vecQuestions)
        if cfg.OUT_QUESTION_MUL:
            features = torch.cat([x_att, eQ, x_att*eQ], dim=-1)
        else:
            features = torch.cat([x_att, eQ], dim=-1)
        logits = self.classifier_layer(features)
        return logits


class BboxRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.bbox_offset_fcn = ops.Linear(cfg.CTX_DIM, 4)

    def forward(self, x_out, ref_scores):
        bbox_offset_fcn = self.bbox_offset_fcn(x_out)
        max_inds = torch.argmax(ref_scores, dim=1, keepdim=True).squeeze()
        probabilities = torch.zeros_like(ref_scores)
        probabilities[torch.arange(len(max_inds), dtype=int), max_inds] = 1
        #BRYCE CODE
        #print('BboxRegression')
        #print('bbox_offset_fcn: ', bbox_offset_fcn.shape)
        assert len(x_out.size()) == 3
        slice_inds = np.argwhere(probabilities.detach().cpu().numpy() == 1)
        #print('slice_inds: ', slice_inds.shape)
        #print(slice_inds)
        bbox_offset = bbox_offset_fcn[slice_inds[:,0], slice_inds[:,1], :]
        #print('bbox_offset: ', bbox_offset.shape)
        #print(bbox_offset)

        return bbox_offset, bbox_offset_fcn, slice_inds
        #BRYCE CODE
