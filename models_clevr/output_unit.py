import torch
from torch import nn

from . import ops as ops
from .config import cfg
import numpy as np

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
        # fcn calculates the bounding box offsets for all boxes
        bbox_offset_fcn = self.bbox_offset_fcn(x_out)
        # probabilities is the same as ref_scores except that the maximum probability is always above threshold
        probabilities = ref_scores.clone()
        max_inds = torch.argmax(probabilities, dim=1).squeeze()
        probabilities[torch.arange(probabilities.shape[0]), max_inds] = 1
        #slice inds is the indices where the probability is above threshold (or is the maximum probability).  These are the predicted indices
        slice_inds = (probabilities > cfg.MATCH_THRESH).nonzero()
        #bbox offset is the offset for the predicted boxes
        bbox_offset = bbox_offset_fcn[slice_inds[:,0], slice_inds[:,1], :]
        return bbox_offset, bbox_offset_fcn, slice_inds
