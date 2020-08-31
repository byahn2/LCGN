from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from . import ops as ops
from .config import cfg
from .lcgn import LCGN
from .input_unit import Encoder
from .output_unit import Classifier, BboxRegression

import time
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import precision_recall_curve, f1_score, auc
import matplotlib.pyplot as plt

from util.boxes import batch_feat_grid2bbox, batch_bbox_iou


class SingleHop(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_q = ops.Linear(cfg.ENC_DIM, cfg.CTX_DIM)
        self.inter2att = ops.Linear(cfg.CTX_DIM, 1)

    def forward(self, kb, vecQuestions, imagesObjectNum):
        proj_q = self.proj_q(vecQuestions)
        interactions = F.normalize(kb * proj_q[:, None, :], dim=-1)
        raw_att = self.inter2att(interactions).squeeze(-1)
        raw_att = ops.apply_mask1d(raw_att, imagesObjectNum)
        att = F.softmax(raw_att, dim=-1)

        x_att = torch.bmm(att[:, None, :], kb).squeeze(1)
        return x_att


class GroundeR(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_q = ops.Linear(cfg.ENC_DIM, cfg.CTX_DIM)
        self.inter2att = ops.Linear(cfg.CTX_DIM, 1)

    def forward(self, kb, vecQuestions, imagesObjectNum):
        proj_q = self.proj_q(vecQuestions)
        interactions = F.normalize(kb * proj_q[:, None, :], dim=-1)
        logits = self.inter2att(interactions).squeeze(-1)
        logits = ops.apply_mask1d(logits, imagesObjectNum)
        return logits


class LCGNnet(nn.Module):
    def __init__(self, num_vocab, num_choices):
        super().__init__()
        if cfg.INIT_WRD_EMB_FROM_FILE:
            embeddingsInit = np.load(cfg.WRD_EMB_INIT_FILE)
            assert embeddingsInit.shape == (num_vocab-1, cfg.WRD_EMB_DIM)
        else:
            embeddingsInit = np.random.uniform(
                low=-1, high=1, size=(num_vocab-1, cfg.WRD_EMB_DIM))
        self.num_vocab = num_vocab
        self.num_choices = num_choices
        self.encoder = Encoder(embeddingsInit)
        self.lcgn = LCGN()
        if cfg.BUILD_VQA:
            self.single_hop = SingleHop()
            self.classifier = Classifier(num_choices)
        if cfg.BUILD_REF:
            self.grounder = GroundeR()
            self.bbox_regression = BboxRegression()

    def forward(self, batch, run_vqa, run_ref):
        batchSize = len(batch['image_feat_batch'])
        questionIndices = torch.from_numpy(
            batch['input_seq_batch'].astype(np.int64)).cuda()
        questionLengths = torch.from_numpy(
            batch['seq_length_batch'].astype(np.int64)).cuda()
        images = torch.from_numpy(
            batch['image_feat_batch'].astype(np.float32)).cuda()
        imagesObjectNum = torch.from_numpy(
            np.sum(batch['image_valid_batch'].astype(np.int64), axis=1)).cuda()
        if run_vqa:
            answerIndices = torch.from_numpy(
                batch['answer_label_batch'].astype(np.int64)).cuda()
        if run_ref:
            bboxInd = batch['bbox_ind_batch']
            bboxOffset = batch['bbox_offset_batch'] 
            bboxCoords = batch['bbox_batch'] 
            
            # initialize zero matricies for each object in each batch to store target information (these are batch_size x 20 x 4)
            bboxRefScoreGt = torch.zeros(size=(batchSize, torch.max(imagesObjectNum)), dtype=torch.float64).cuda()
            bboxOffsetGt = torch.zeros(size=(batchSize, torch.max(imagesObjectNum), 4), dtype=torch.float32).cuda()
            bboxCoordsGt = torch.zeros(size=(batchSize, torch.max(imagesObjectNum), 4), dtype = torch.int64).cuda()
            
            # batch_inds is a list of the indices of the batches where there are gt positive boxes: eg: [0, 0, 0, 1, 2, 3, 3, 4, 5, 5, 5 ... 63 63] 
            # box_inds is the index of bboxInd, bboxOffset, and bboxCoords that contain a ground truth box: eg [0, 1 ,3 , 0, 2 ... 5, 0]
            # target_inds is the indices of the proposed boxes which are gt positive eg [140, 28, 32, 49, 160, ... 2]
            batch_inds = np.argwhere(bboxInd > -1)[:,0]
            box_inds = np.argwhere(bboxInd > -1)[:,1]
            target_inds = bboxInd[bboxInd > -1]
            
            bboxRefScoreGt[batch_inds[:], target_inds[:]] = 1
            bboxOffsetGt[batch_inds[:], target_inds[:], :] = torch.from_numpy(bboxOffset[batch_inds[:], box_inds[:], :].astype(np.float32)).cuda()
            bboxCoordsGt[batch_inds[:], target_inds[:], :] = torch.from_numpy(bboxCoords[batch_inds[:], box_inds[:], :].astype(np.int64)).cuda()
        
        # LSTM
        questionCntxWords, vecQuestions = self.encoder(
            questionIndices, questionLengths)
        
        # LCGN
        x_out = self.lcgn(
            images=images, q_encoding=vecQuestions,
            lstm_outputs=questionCntxWords, batch_size=batchSize,
            q_length=questionLengths, entity_num=imagesObjectNum)
        
        # Single-Hop
        loss = torch.tensor(0., device=x_out.device)
        res = {}
        if run_vqa:
            x_att = self.single_hop(x_out, vecQuestions, imagesObjectNum)
            logits = self.classifier(x_att, vecQuestions)
            predictions, num_correct = self.add_pred_op(logits, answerIndices)
            loss += self.add_answer_loss_op(logits, answerIndices)
            res.update({
                "predictions": predictions,
                "num_correct": int(num_correct),
                "accuracy": float(num_correct * 1. / batchSize)
            })

        if run_ref:
            assert cfg.FEAT_TYPE == 'spatial'
            #calculate ref_scores
            ref_scores = torch.sigmoid(self.grounder(x_out, vecQuestions, imagesObjectNum))
           
            # calculate bbox_offset (this was not trained)
            bbox_offset, bbox_offset_fcn, ref_inds = self.bbox_regression(x_out, ref_scores)
            # bbox predictions returns a matrix that is batch_size x num_boxes x 4.  
            # It has the predicted x,y,w,h of all bounding boxes with matching scores higher than the threshold, all other coordinates are 0 
            bbox_predictions = batch_feat_grid2bbox(ref_inds.detach().cpu().numpy(), bboxCoordsGt.shape, bbox_offset.detach().cpu().numpy(), cfg.IMG_H / cfg.H_FEAT, cfg.IMG_W / cfg.W_FEAT,cfg.H_FEAT, cfg.W_FEAT)

            #calculate the loss
            bbox_ind_loss, bbox_offset_loss = self.add_bbox_loss_op(ref_scores, bbox_offset_fcn, bboxRefScoreGt, bboxOffsetGt)
            loss += (bbox_ind_loss + bbox_offset_loss)
             
            # for normal version, calculate box ious to use as a metric
            bbox_ious = batch_bbox_iou(bbox_predictions, bboxCoordsGt, bboxRefScoreGt)
            bbox_num_correct = np.sum(bbox_ious >= cfg.BBOX_IOU_THRESH)
            possible_correct_boxes = torch.sum(bboxRefScoreGt).item()

            # calculate number of positives, negatives, and AUC using function
            true_positive, total_positive, true_negative, total_negative, precision, top_accuracy_list = self.calc_correct(bboxRefScoreGt, ref_scores)

            possible_correct = float(bboxRefScoreGt.shape[0]*bboxRefScoreGt.shape[1])
            res.update({
                "top_accuracy_list" : top_accuracy_list.detach().cpu().numpy().astype(float),
                "bbox_predictions": bbox_predictions.astype(float),
                "gt_coords": bboxCoordsGt.detach().cpu().numpy().astype(float),
                "bbox_ious": bbox_ious.astype(float),
                "true_positive": int(true_positive),
                "true_negative": int(true_negative),
                "false_positive": int(total_negative-true_negative),
                "false_negative": int(total_positive - true_positive),
                "bbox_num_correct": int(bbox_num_correct),
                "num_correct": int(true_positive + true_negative),
                "top_accuracy": float(np.mean(top_accuracy_list.detach().cpu().numpy())),
                "bbox_accuracy": float((bbox_num_correct * 1.)/possible_correct_boxes),
                "possible_correct": float(possible_correct),
                "possible_correct_boxes": int(possible_correct_boxes),
                "precision": float(precision),
            })
        res.update({"batch_size": int(batchSize), "loss": loss})
        return res

    def calc_correct(self, gt_scores, ref_scores_original):
        batchSize = gt_scores.shape[0]
        # ref_scores is the same as the original ref_scores, but the top probability is always above threshold
        ref_scores = ref_scores_original.clone() 
        max_inds = torch.argmax(ref_scores, dim=1).squeeze()
        ref_scores[torch.arange(ref_scores.shape[0]), max_inds] = 1

        # slice inds is the indices where the ground truth positives are
        slice_inds = (gt_scores !=0).nonzero()
        total_positive = slice_inds.shape[0]
        ref_slice = ref_scores[slice_inds[:,0], slice_inds[:,1]]
        # slice_inds_neg is the indices where the ground truth negatives are
        slice_inds_neg = (gt_scores == 0).nonzero()
        total_negative = slice_inds_neg.shape[0]
        ref_slice_neg = ref_scores[slice_inds_neg[:,0], slice_inds_neg[:,1]]
        #the means of the values for the probabilities at gt positive and gt negative indiceis
        pos_mean = torch.mean(ref_slice, dim=0)
        neg_mean = torch.mean(ref_slice_neg, dim=0)

        # thresh classification is the actual classification of each box based on the threshold, 0 for predicted negative, 1 for predicted positive
        thresh_classifications = ref_scores.clone()
        thresh_classifications[thresh_classifications >= cfg.MATCH_THRESH] = 1
        thresh_classifications[thresh_classifications < cfg.MATCH_THRESH] = 0
        true_positive = len((ref_slice >= cfg.MATCH_THRESH).nonzero())
        true_negative = len((ref_slice_neg < cfg.MATCH_THRESH).nonzero())
        false_negative = total_positive - true_positive
        false_positive = total_negative - true_negative
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)

        # calculate top positive
        num_gt_pos =torch.sum(gt_scores, dim=1).int()
        top_pos = torch.sum((gt_scores * thresh_classifications), dim=1).float()
        binary_top_accuracy = top_pos / num_gt_pos
        top_accuracy = torch.zeros(batchSize)
        for b in range(batchSize):
            top_k, top_k_inds = torch.topk(ref_scores[b,:], k=num_gt_pos[b].item())
            num_correct = torch.sum(gt_scores[(b*torch.ones(len(top_k_inds),dtype=int).cuda()), top_k_inds])
            top_accuracy[b] = num_correct/num_gt_pos[b]

        # recalculate for thresh in config = 0.9 and return results
        print('Precisions: ', precision)
        print('Recall: ', recall)
        print('TRUE POSITIVE: ', true_positive, ' FALSE POSITIVE: ', false_positive)
        print('TRUE_NEGATIVE: ', true_negative, 'FALSE_NEGATIVE: ', false_negative)
        print('CORRECT: ', true_negative + true_positive, ' INCORRECT: ', gt_scores.shape[0]*gt_scores.shape[1]-(true_negative + true_positive))
        print('Top Accuracy: ', torch.mean(top_accuracy))
        return (true_positive, total_positive, true_negative, total_negative, precision, top_accuracy)#, AUC, f1)

    def add_pred_op(self, logits, answers):
        if cfg.MASK_PADUNK_IN_LOGITS:
            logits = logits.clone()
            logits[..., :2] += -1e30  # mask <pad> and <unk>

        preds = torch.argmax(logits, dim=-1).detach()
        corrects = (preds == answers)
        correctNum = torch.sum(corrects).item()
        preds = preds.cpu().numpy()

        return preds, correctNum

    def add_answer_loss_op(self, logits, answers):
        if cfg.TRAIN.LOSS_TYPE == "softmax":
            loss = F.cross_entropy(logits, answers)
        elif cfg.TRAIN.LOSS_TYPE == "sigmoid":
            answerDist = F.one_hot(answers, self.num_choices).float()
            loss = F.binary_cross_entropy_with_logits(
                logits, answerDist) * self.num_choices
        else:
            raise Exception("non-identified loss")
        return loss

    def add_bbox_loss_op(self, ref_scores, bbox_offset_fcn, bbox_ind_gt, bbox_offset_gt):
        # bounding box selection loss
        #calculate weights for unbalanced dataset
        gt_positive = 1.0 * torch.sum(bbox_ind_gt).item()
        gt_negative = 1.0 * (bbox_ind_gt.shape[0] * bbox_ind_gt.shape[1] - gt_positive)
        gt_total = 1.0 * bbox_ind_gt.shape[0]*bbox_ind_gt.shape[1]
        positive_weight = 1 - (gt_positive/gt_total)
        negative_weight = 1 - (gt_negative/gt_total)
        weight_matrix = (((positive_weight-negative_weight) * bbox_ind_gt) + negative_weight).cuda()
        bbox_ind_loss = F.binary_cross_entropy_with_logits(input=ref_scores, target=bbox_ind_gt, weight=weight_matrix, reduction='mean')

        # bounding box regression loss
        slice_inds = (bbox_ind_gt != 0).nonzero()
        ref_scores_sliced = ref_scores[slice_inds[:,0], slice_inds[:,1]]
        bbox_ind_gt_sliced = bbox_ind_gt[slice_inds[:,0], slice_inds[:,1]]
        bbox_offset_sliced = bbox_offset_fcn[slice_inds[:,0], slice_inds[:,1], :]
        gt_offset_sliced = bbox_offset_gt[slice_inds[:,0], slice_inds[:,1], :]
        bbox_offset_loss = F.mse_loss(bbox_offset_sliced, gt_offset_sliced)
        
        return bbox_ind_loss, bbox_offset_loss


class LCGNwrapper():
    def __init__(self, num_vocab, num_choices):
        self.model = LCGNnet(num_vocab, num_choices).cuda()
        self.trainable_params = [
            p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(
            self.trainable_params, lr=cfg.TRAIN.SOLVER.LR)
        self.lr = cfg.TRAIN.SOLVER.LR

        if cfg.USE_EMA:
            self.ema_param_dict = {
                name: p for name, p in self.model.named_parameters()
                if p.requires_grad}
            self.ema = ops.ExponentialMovingAverage(
                self.ema_param_dict, decay=cfg.EMA_DECAY_RATE)
            self.using_ema_params = False

    def train(self, training=True):
        self.model.train(training)
        if training:
            self.set_params_from_original()
        else:
            self.set_params_from_ema()

    def eval(self):
        self.train(False)

    def state_dict(self):
        # Generate state dict in training mode
        current_mode = self.model.training
        self.train(True)

        assert (not cfg.USE_EMA) or (not self.using_ema_params)
        return {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'ema': self.ema.state_dict() if cfg.USE_EMA else None
        }

        # restore original mode
        self.train(current_mode)

    def load_state_dict(self, state_dict):
        # Load parameters in training mode
        current_mode = self.model.training
        self.train(True)

        assert (not cfg.USE_EMA) or (not self.using_ema_params)
        self.model.load_state_dict(state_dict['model'])

        if 'optimizer' in state_dict:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        else:
            print('Optimizer does not exist in checkpoint! '
                  'Loaded only model parameters.')

        if cfg.USE_EMA:
            if 'ema' in state_dict and state_dict['ema'] is not None:
                self.ema.load_state_dict(state_dict['ema'])
            else:
                print('cfg.USE_EMA is True, but EMA does not exist in '
                      'checkpoint! Using model params to initialize EMA.')
                self.ema.load_state_dict(
                    {k: p.data for k, p in self.ema_param_dict.items()})

        # restore original mode
        self.train(current_mode)

    def set_params_from_ema(self):
        if (not cfg.USE_EMA) or self.using_ema_params:
            return

        self.original_state_dict = deepcopy(self.model.state_dict())
        self.ema.set_params_from_ema(self.ema_param_dict)
        self.using_ema_params = True

    def set_params_from_original(self):
        if (not cfg.USE_EMA) or (not self.using_ema_params):
            return

        self.model.load_state_dict(self.original_state_dict)
        self.using_ema_params = False

    def run_batch(self, batch, train, run_vqa, run_ref, lr=None):
        assert train == self.model.training
        assert (not train) or (lr is not None), 'lr must be set for training'

        if train:
            if lr != self.lr:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                self.lr = lr
            self.optimizer.zero_grad()
            batch_res = self.model.forward(batch, run_vqa, run_ref)
            loss = batch_res['loss']
            loss.backward()
            if cfg.TRAIN.CLIP_GRADIENTS:
                nn.utils.clip_grad_norm_(
                    self.trainable_params, cfg.TRAIN.GRAD_MAX_NORM)
            self.optimizer.step()
            if cfg.USE_EMA:
                self.ema.step(self.ema_param_dict)
        else:
            with torch.no_grad():
                batch_res = self.model.forward(batch, run_vqa, run_ref)
        return batch_res
