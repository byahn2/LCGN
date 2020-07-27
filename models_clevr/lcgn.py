import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from . import ops as ops
from .config import cfg


class LCGN(nn.Module):
    # LCGN model is composed of the context initiator, textual command extractor, and message passing
    def __init__(self):
        super().__init__()
        self.build_loc_ctx_init()
        self.build_extract_textual_command()
        self.build_propagate_message()

    # the context initiator learns how to initialize the context representations for the first iteration t=0
    def build_loc_ctx_init(self):
        assert cfg.STEM_LINEAR != cfg.STEM_CNN
        if cfg.STEM_LINEAR:
            self.initKB = ops.Linear(cfg.D_FEAT, cfg.CTX_DIM) #D_FEAT = feature dimensions = 1152 = 1024+128 CTX_DIM = context dimensions = 512
            self.x_loc_drop = nn.Dropout(1 - cfg.stemDropout) #stemDropout = 1
        elif cfg.STEM_CNN:
            self.cnn = nn.Sequential(
                nn.Dropout(1 - cfg.stemDropout),
                ops.Conv(cfg.D_FEAT, cfg.STEM_CNN_DIM, (3, 3), padding=1), #STEM_CNN_DIM = 512
                nn.ELU(),
                nn.Dropout(1 - cfg.stemDropout),
                ops.Conv(cfg.STEM_CNN_DIM, cfg.CTX_DIM, (3, 3), padding=1), #STEM CNN_DIM = 512, CTX_DIM = 512
                nn.ELU())

        self.initMem = nn.Parameter(torch.randn(1, 1, cfg.CTX_DIM))

    # textual command extraction learns a set of t commands from the input text
    def build_extract_textual_command(self):
        #self.qInput = W3 from equation 2
        self.qInput = ops.Linear(cfg.CMD_DIM, cfg.CMD_DIM) #CMD_DIM = 512
        for t in range(cfg.MSG_ITER_NUM): #MSG_ITR_NUM = 4
            #qInput_layer2 is W2(t) from equation 2
            qInput_layer2 = ops.Linear(cfg.CMD_DIM, cfg.CMD_DIM)
            setattr(self, "qInput%d" % t, qInput_layer2)
        # self.cmd_inter2logits = W1 from equation 2
        self.cmd_inter2logits = ops.Linear(cfg.CMD_DIM, 1)

    
    def build_propagate_message(self):
        self.read_drop = nn.Dropout(1 - cfg.readDropout) #readDropout = 0.85
        #self.project_x_loc = W4 from equation 4
        self.project_x_loc = ops.Linear(cfg.CTX_DIM, cfg.CTX_DIM) #CTX_DIM = 512
        #self.project_x_ctx = W5 from equation 4
        self.project_x_ctx = ops.Linear(cfg.CTX_DIM, cfg.CTX_DIM)
        #self.queries = W6 from equation 5
        self.queries = ops.Linear(3*cfg.CTX_DIM, cfg.CTX_DIM)
        #self.keys = W7 from equation 5
        self.keys = ops.Linear(3*cfg.CTX_DIM, cfg.CTX_DIM)
        #self.vals = W9 from equation 5
        self.vals = ops.Linear(3*cfg.CTX_DIM, cfg.CTX_DIM)
        #self.proj_keys = W8 from equation 5
        self.proj_keys = ops.Linear(cfg.CMD_DIM, cfg.CTX_DIM) #CMD_DIM = 512
        #self.proj_values = W10 from equation 5
        self.proj_vals = ops.Linear(cfg.CMD_DIM, cfg.CTX_DIM)
        #self.mem_update = W11 from equation 7
        self.mem_update = ops.Linear(2*cfg.CTX_DIM, cfg.CTX_DIM)
        #self.combine_kb = W12 from equation 8
        self.combine_kb = ops.Linear(2*cfg.CTX_DIM, cfg.CTX_DIM)


    # retrieves the local feature information and initializes the context information
    # for t iterations, updates the context for each region using message passing
    # outputs the concatenation of the local feature information and the context information after t iterations
    def forward(self, images, q_encoding, lstm_outputs, batch_size, q_length,
                entity_num):
        x_loc, x_ctx, x_ctx_var_drop = self.loc_ctx_init(images)
        for t in range(cfg.MSG_ITER_NUM): #MSG_ITER_NUM = 4
            x_ctx = self.run_message_passing_iter(
                q_encoding, lstm_outputs, q_length, x_loc, x_ctx,
                x_ctx_var_drop, entity_num, t)
        x_out = self.combine_kb(torch.cat([x_loc, x_ctx], dim=-1)) # self.combine_kb is W12 from equation 8
        return x_out

    # takes the lstm outputs, q_length = number of words, t = iteration, and q encoding which is the summary vector of lstm outputs
    # calcualtes the attention and returns the command c_t for iteration t
    def extract_textual_command(self, q_encoding, lstm_outputs, q_length, t):
        qInput_layer2 = getattr(self, "qInput%d" % t)
        act_fun = ops.activations[cfg.CMD_INPUT_ACT] #CMD_INPUT_ACT = 'ELU'
        q_cmd = qInput_layer2(act_fun(self.qInput(q_encoding))) #self.qInput is W3 and qInput_layer2 is W2(t) frome equation 2
        raw_att = self.cmd_inter2logits(
            q_cmd[:, None, :] * lstm_outputs).squeeze(-1) #self.cmd_inter2logits is W1 from equation 2
        raw_att = ops.apply_mask1d(raw_att, q_length)
        att = F.softmax(raw_att, dim=-1)
        cmd = torch.bmm(att[:, None, :], lstm_outputs).squeeze(1) #Equation 1
        # torch.bmm performs a batch matrix-matrix product
        return cmd

    # run for one entity
    # calculates x joint from local features and context features for t-1
    # what are queries, keys, and vals?
    # calculates edge weights
    # calculates messages
    #updates context by concatenating incoming messages to update context representation
    def propagate_message(self, cmd, x_loc, x_ctx, x_ctx_var_drop, entity_num):
        x_ctx = x_ctx * x_ctx_var_drop
        proj_x_loc = self.project_x_loc(self.read_drop(x_loc)) #proj_x_loc = W4*x_loc from equation 4
        proj_x_ctx = self.project_x_ctx(self.read_drop(x_ctx)) #proj_x_ctx = W5*x_ctx from equation 4
        x_joint = torch.cat(
            [x_loc, x_ctx, proj_x_loc * proj_x_ctx], dim=-1)

        queries = self.queries(x_joint) #queries ar W6*x_joint from equation 5
        keys = self.keys(x_joint) * self.proj_keys(cmd)[:, None, :] #keys = (W7*x_joint) * (W8*cmd)) from eq 5
        vals = self.vals(x_joint) * self.proj_vals(cmd)[:, None, :] #vals are ((W9*x_joint) * (W10*cmd)) from equation 6
        edge_score = (
            torch.bmm(queries, torch.transpose(keys, 1, 2)) /
            np.sqrt(cfg.CTX_DIM))
        edge_score = ops.apply_mask2d(edge_score, entity_num)
        edge_prob = F.softmax(edge_score, dim=-1)
        message = torch.bmm(edge_prob, vals) #message is weight * keys from equation 6

        x_ctx_new = self.mem_update(torch.cat([x_ctx, message], dim=-1)) #new_ctx = W11*(old_ctx * sum of messages) from equation 7
        return x_ctx_new

    # for one iteration, extracts textual command and updates context
    def run_message_passing_iter(
            self, q_encoding, lstm_outputs, q_length, x_loc, x_ctx,
            x_ctx_var_drop, entity_num, t):
        cmd = self.extract_textual_command(
                q_encoding, lstm_outputs, q_length, t)
        x_ctx = self.propagate_message(
            cmd, x_loc, x_ctx, x_ctx_var_drop, entity_num)
        return x_ctx

    # ititializes context
    def loc_ctx_init(self, images):
        if cfg.STEM_NORMALIZE: #STEM_NORMALIZE = True
            images = F.normalize(images, dim=-1)
        if cfg.STEM_LINEAR: #STEM_LINEAR = True
            x_loc = self.initKB(images)
            x_loc = self.x_loc_drop(x_loc)
        elif cfg.STEM_CNN:
            images = torch.transpose(images, 1, 2)  # N(HW)C => NC(HW)
            x_loc = images.view(-1, cfg.D_FEAT, cfg.H_FEAT, cfg.W_FEAT)
            x_loc = self.cnn(x_loc)
            x_loc = x_loc.view(-1, cfg.CTX_DIM, cfg.H_FEAT * cfg.W_FEAT)
            x_loc = torch.transpose(x_loc, 1, 2)  # NC(HW) => N(HW)C
        if cfg.STEM_RENORMALIZE: #STEM RENORMALIZE = False
            x_loc = F.normalize(x_loc, dim=-1)

        x_ctx = self.initMem.expand(x_loc.size())
        x_ctx_var_drop = ops.generate_scaled_var_drop_mask(
            x_ctx.size(),
            keep_prob=(cfg.memoryDropout if self.training else 1.))
        return x_loc, x_ctx, x_ctx_var_drop
