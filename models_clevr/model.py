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
        #groundR_time = time.time()
        proj_q = self.proj_q(vecQuestions)
        interactions = F.normalize(kb * proj_q[:, None, :], dim=-1)
        logits = self.inter2att(interactions).squeeze(-1)
        logits = ops.apply_mask1d(logits, imagesObjectNum)
        #print('GroundR time: ', time.time()-groundR_time)
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
        #forward_time = time.time()
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
        #BRYCE CODE
        build_gt_time = time.time()
        if run_ref:
            # get gt ind, offset, and coordinates from batch 
            bboxInd = batch['bbox_ind_batch'] #torch.from_numpy(batch['bbox_ind_batch'].astype(np.int64)).cuda()
            bboxOffset = batch['bbox_offset_batch'] #torch.from_numpy(batch['bbox_offset_batch'].astype(np.float32)).cuda()
            bboxBatch = batch['bbox_batch'] #torch.from_numpy(batch['bbox_batch'].astype(np.float32)).cuda()
            #print('initial bboxInd: ', bboxInd.shape)
            #print('initial bboxOffset: ', bboxOffset.shape)
            #print('initial bboxBatch: ', bboxBatch.shape)
            # initialize zero matricies for each object in each batch to store target information
            bboxRefScoreGt = torch.zeros(size=(batchSize, torch.max(imagesObjectNum)), dtype=torch.float64).cuda()
            bboxOffsetGt = torch.zeros(size=(batchSize, torch.max(imagesObjectNum), 4), dtype=torch.float32).cuda()
            bboxBatchGt = torch.zeros(size=(batchSize, torch.max(imagesObjectNum), 4), dtype = torch.int64).cuda()
            # for each batch and each possible target, check if there is a target index or a negative number
            # if there is a target index, that will be the object index for a target
            # set the bbox_RefScore at that index to 1 and set the offset and batch GT at that index to the respective offset and bbox coordinates
            batch_inds = np.argwhere(bboxInd > -1)[:,0]
            box_inds = np.argwhere(bboxInd > -1)[:,1]
            target_inds = bboxInd[bboxInd > -1]
            #print('batch_inds: ', len(batch_inds), ' ', batch_inds)
            #print('box_inds: ', len(box_inds), ' ', box_inds)
            #print('target_inds: ', len(target_inds), ' ', target_inds)
            bboxRefScoreGt[batch_inds[:], target_inds[:]] = 1
            bboxOffsetGt[batch_inds[:], target_inds[:], :] = torch.from_numpy(bboxOffset[batch_inds[:], box_inds[:], :].astype(np.float32)).cuda()
            bboxBatchGt[batch_inds[:], target_inds[:], :] = torch.from_numpy(bboxBatch[batch_inds[:], box_inds[:], :].astype(np.int64)).cuda()

            #print('bboxRefScoreGt: ', bboxRefScoreGt.shape)
            #print('bboxOffsetGt: ', bboxOffsetGt.shape)
            #print('bboxBatchGt: ', bboxBatchGt.shape)
        #BRYCE CODE
        print('build_gt_time: ', time.time() - build_gt_time)
        LSTM_time = time.time()

        # LSTM
        questionCntxWords, vecQuestions = self.encoder(
            questionIndices, questionLengths)
        print('LSTM_Time: ', time.time() - LSTM_time)
        LCGN_time = time.time()
        # LCGN
        x_out = self.lcgn(
            images=images, q_encoding=vecQuestions,
            lstm_outputs=questionCntxWords, batch_size=batchSize,
            q_length=questionLengths, entity_num=imagesObjectNum)
        print('LCGN_Time: ', time.time() - LCGN_time)
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
            ref_scores_time = time.time()
            assert cfg.FEAT_TYPE == 'spatial'
            ref_scores = self.grounder(x_out, vecQuestions, imagesObjectNum)
            #print('ref_scores_time: ', time.time()-ref_scores_time)
            #BRYCE CODE
            #print('ref_scores from groundr: ', ref_scores.shape)
            
            bbox_offset, bbox_offset_fcn, ref_inds = self.bbox_regression(x_out, ref_scores)
            
            #print('bbox_offset: ', bbox_offset.shape)
            #print('bbox_offset_fcn: ', bbox_offset_fcn.shape)
            #print('ref_inds: ', ref_inds.shape)
            # bbox predictions returns a matrix that is batch_size x num_boxes x 4.  
            # It has the predicted x,y,w,h of all bounding boxes with matching scores higher than the threshold 
            
            bbox_predictions = batch_feat_grid2bbox(ref_inds, bboxBatchGt.shape,bbox_offset.detach().cpu().numpy(),cfg.IMG_H / cfg.H_FEAT, cfg.IMG_W / cfg.W_FEAT,cfg.H_FEAT, cfg.W_FEAT)
            
            #print('bbox_predictions: ', bbox_predictions[slice_inds[:,0], slice_inds[:,1], :])
            
            #print(bbox_predictions[ref_inds[0,0], ref_inds[0,1], :])
            #print(bboxBatchGt[ref_inds[0,0], ref_inds[0,1], :])
            #DEBUG
            #bbox_ind_loss, bbox_offset_loss = self.add_bbox_loss_op(ref_scores, bbox_offset_fcn, bboxRefScoreGt, bboxOffsetGt)
            loss_time = time.time()
            bbox_ind_loss = self.add_bbox_loss_op(ref_scores, bboxRefScoreGt)
            #DEBUG
            loss += (bbox_ind_loss) #DEBUG  + bbox_offset_loss) #DEBUG
            #print('loss_time: ', time.time()-loss_time)
            #bbox_ious = batch_bbox_iou(bbox_predictions, bboxBatchGt, bboxRefScoreGt)
            
            #print('bbox_ious: ', bbox_ious.shape)
            
            #DEBUG
            #bbox_num_correct = np.sum(bbox_ious >= cfg.BBOX_IOU_THRESH)
            
            true_positive, total_positive, true_negative, total_negative, AUC = self.calc_correct(bboxRefScoreGt, ref_scores)
            
            probabilities = torch.sigmoid(ref_scores)
            probabilities[probabilities > cfg.MATCH_THRESH] = 1
            probabilities[probabilities < cfg.MATCH_THRESH] = 0
            probabilities = torch.abs(bboxRefScoreGt - probabilities)
            accuracy_list = 1 - (torch.sum(probabilities, axis=1) / probabilities.shape[1]).detach().cpu().numpy().astype(float)
            #print('accuracy_list: ', accuracy_list.shape, ' ', accuracy_list)

            # anywhere where we have the ground truth bbox coordinates, replace the bbox predictions with the ground truth
            slice_inds = (bboxRefScoreGt !=0).nonzero()
            bbox_predictions = torch.from_numpy(bbox_predictions).cuda()
            bbox_predictions[slice_inds[:,0], slice_inds[:,1], :] = bboxBatchGt[slice_inds[:,0], slice_inds[:,1], :].double()
            #print('bbox_predictions model: ', bbox_predictions[slice_inds[:,0], slice_inds[:,1],:])
            # only keep the ground thruth box coordinates for boxes with probabilities above threshold
            thresh_inds = (probabilities != 0).nonzero()
            thresh_boxes = bbox_predictions[thresh_inds[:,0], thresh_inds[:,1], :]
            thresh_boxes[:, 2:4][thresh_boxes[:,2:4]<5] = 20
            bbox_predictions[thresh_inds[:,0], thresh_inds[:,1], :] = thresh_boxes
            #print(bbox_predictions[thresh_inds[:,0], thresh_inds[:,1],:])
            bbox_predictions = bbox_predictions.detach().cpu().numpy().astype(int)
            #pause
            #bbox_predictions = bbox_predictions[slice_inds[:,0], slice_inds[:,1], :].detach().cpu().numpy().astype(int)
            
            #print('bbox_predictions_above threshold: ', bbox_predictions)
            #print('bbox_predictions in model.py: ', bbox_predictions.shape)

            #DEBUG
            #possible_correct = torch.sum(bboxRefScoreGt).item()
            res_update_time = time.time()
            possible_correct = float(bboxRefScoreGt.shape[0]*bboxRefScoreGt.shape[1])
            res.update({
                "accuracy_list" : accuracy_list,
                "bbox_predictions": bbox_predictions,
                "gt_coords": bboxBatchGt,
                #"bbox_ious": bbox_ious,
                "true_positive": int(true_positive),
                "true_negative": int(true_negative),
                "false_positive": int(total_negative-true_negative),
                "false_negative": int(total_positive - true_positive),
                "bbox_num_correct": int(true_positive + true_negative),
                "bbox_accuracy": float((true_positive + true_negative) * 1. / possible_correct),
                "possible_correct": float(possible_correct),
                "AUC": float(AUC)
                #"possible_correct": possible_correct
            #BRYCE CODE
            })
        res.update({"batch_size": int(batchSize), "loss": loss})
        #print('res_update_time: ', time.time() - res_update_time)
        #print('forward time: ', time.time() - forward_time)
        return res

    #BRYCE CODE
    def calc_correct(self, gt_scores, ref_scores):
        calc_correct_time = time.time()
        slice_inds = (gt_scores !=0).nonzero()
        total_positive = slice_inds.shape[0]
        ref_slice = ref_scores[slice_inds[:,0], slice_inds[:,1]]
        slice_inds_neg = (gt_scores == 0).nonzero()
        total_negative = slice_inds_neg.shape[0]
        ref_slice_neg = ref_scores[slice_inds_neg[:,0], slice_inds_neg[:,1]]
        
        pos_mean = np.mean(torch.sigmoid(ref_slice).detach().cpu().numpy(), axis=0)
        neg_mean = np.mean(torch.sigmoid(ref_slice_neg).detach().cpu().numpy(), axis=0)
        print('\n Pos Mean: ', pos_mean, ' Neg mean: ', neg_mean)
        
        #ROC
        num_boxes = ref_scores.shape[1]
        batch_size = ref_scores.shape[0]
        probabilities = torch.sigmoid(ref_scores).detach().cpu().numpy()
        #positive = 0 + np.sum(probabilities >= .9)
        gt = gt_scores.detach().cpu().numpy()
        if batch_size == 1:
            gt = np.expand_dims(gt, axis=0)
            probabilities = np.expand_dims(probabilities, axis=0)
        AUC = 0
        for b in range(batch_size):
            fpr = dict() 
            tpr = dict()
            roc_auc = dict()
            fpr[b], tpr[b], _ = roc_curve(gt[b,:].T, probabilities[b,:].T)
            roc_auc[b] = auc(fpr[b], tpr[b])
            AUC += roc_auc[b]
            print('\nauc is ', roc_auc[b])
        AUC = AUC / batch_size
        
        for thresh in np.arange(0, 1.01, 0.05):
            true_positive = np.sum(torch.sigmoid(ref_slice).detach().cpu().numpy() >= thresh)
            true_negative = np.sum(torch.sigmoid(ref_slice_neg).detach().cpu().numpy() < thresh)
            false_negative = total_positive - true_positive
            false_positive = total_negative - true_negative
            precision = true_positive / (true_positive + false_positive)
            recall = true_positive / (true_positive + false_negative)
            print('\n\n Threshold: ', thresh)
            print('Precisions: ', precision)
            print('Recall: ', recall)
            print('TRUE POSITIVE: ', true_positive, ' FALSE POSITIVE: ', total_negative - true_negative)
            print('CORRECT: ', true_negative + true_positive, ' INCORRECT: ', gt_scores.shape[0]*gt_scores.shape[1]-(true_negative + true_positive))
            print('Accuracy: ', (true_negative + true_positive) / (gt_scores.shape[0]*gt_scores.shape[1]))
        true_positive = np.sum(torch.sigmoid(ref_slice).detach().cpu().numpy() >= cfg.MATCH_THRESH)
        true_negative = np.sum(torch.sigmoid(ref_slice_neg).detach().cpu().numpy() < cfg.MATCH_THRESH)
        
        print('\n\n Threshold: ', cfg.MATCH_THRESH)
        print('Precisions: ', precision)
        print('Recall: ', recall)
        print('TRUE POSITIVE: ', true_positive, ' FALSE POSITIVE: ', total_negative - true_negative)
        print('CORRECT: ', true_negative + true_positive, ' INCORRECT: ', gt_scores.shape[0]*gt_scores.shape[1]-(true_negative + true_positive))
        print('Accuracy: ', (true_negative + true_positive) / (gt_scores.shape[0]*gt_scores.shape[1]))
        #print('calc_correct_time: ', time.time()-calc_correct_time)
        return (true_positive, total_positive, true_negative, total_negative, AUC)

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

    #DEBUG
    #def add_bbox_loss_op(self, ref_scores, bbox_offset_fcn, bbox_ind_gt, bbox_offset_gt):
        #BRYCE CODE
        # bounding box selection loss
        
        #print('ref_scores: ', ref_scores.shape, ' bbox_ind_gt: ', bbox_ind_gt.shape)
    def add_bbox_loss_op(self, ref_scores, bbox_ind_gt):
    #DEBUG
        # Using weight 
        gt_positive = 1.0 * torch.sum(bbox_ind_gt).item()
        gt_negative = 1.0 * (bbox_ind_gt.shape[0] * bbox_ind_gt.shape[1] - gt_positive)
        gt_total = 1.0 * bbox_ind_gt.shape[0]*bbox_ind_gt.shape[1]
        #print('\ngt_positive: ', gt_positive, ' gt_negative: ', gt_negative, ' total: ', gt_positive+gt_negative, ' gt_shape: ', bbox_ind_gt.shape[0]*bbox_ind_gt.shape[1])
        positive_weight = 1 - (gt_positive/gt_total)
        negative_weight = 1 - (gt_negative/gt_total)
        #print('positive_weight: ', positive_weight)
        weight_matrix = (((positive_weight-negative_weight) * bbox_ind_gt) + negative_weight).cuda()
        #print('weight_matrix: ', weight_matrix[0,:])
        #print(weight_matrix[1,:])
        bbox_ind_loss = F.binary_cross_entropy_with_logits(input=ref_scores, target=bbox_ind_gt, weight=weight_matrix, reduction='mean')
         
        #bbox_ind_loss = F.binary_cross_entropy_with_logits(ref_scores, bbox_ind_gt)
        #print('\nref_scores: ', torch.sigmoid(ref_scores).view(-1))
        #print('gt: ', bbox_ind_gt.view(-1))
        #print('gt size: ', bbox_ind_gt.shape)
        # bounding box regression loss
       
        #slice_inds = (bbox_ind_gt != 0).nonzero()
        #print(weight_matrix[slice_inds[:,0], slice_inds[:,1]])

        #ref_scores_sliced = ref_scores[slice_inds[:,0], slice_inds[:,1]]
        #bbox_ind_gt_sliced = bbox_ind_gt[slice_inds[:,0], slice_inds[:,1]]
        #print('ref_scores_sliced: ', torch.sigmoid(ref_scores_sliced))
        #print('max ref_score: ', torch.max(ref_scores).item())

        #print('slice_inds: ', slice_inds.shape)
        
        #bbox_offset_sliced = bbox_offset_fcn[slice_inds[:,0], slice_inds[:,1], :]
        #gt_offset_sliced = bbox_offset_gt[slice_inds[:,0], slice_inds[:,1], :]
        
        #print('bbox_offset_flat: ', bbox_offset_fcn.shape)
        #print('bbox_offset_gt: ', bbox_offset_gt.shape)
        #print('bbox_offset_sliced: ', bbox_offset_sliced.shape)
        #print('gt_offset_sliced: ', gt_offset_sliced.shape)
        
        #bbox_offset_loss = F.mse_loss(bbox_offset_sliced, gt_offset_sliced)
        
        #BRYCE CODE
        return bbox_ind_loss#, bbox_offset_loss


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
            start_time = time.time()
            if lr != self.lr:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                self.lr = lr
            self.optimizer.zero_grad()
            print('start_time: ', time.time()-start_time)
            forward_time = time.time()
            batch_res = self.model.forward(batch, run_vqa, run_ref)
            print('forward_time: ', time.time() - forward_time)
            backward_time = time.time()
            loss = batch_res['loss']
            loss.backward()
            print('backward_time: ', time.time() - backward_time)
            if cfg.TRAIN.CLIP_GRADIENTS:
                clip_grad_time = time.time()
                nn.utils.clip_grad_norm_(
                    self.trainable_params, cfg.TRAIN.GRAD_MAX_NORM)
                print('clip_grad_time: ', time.time()-clip_grad_time)
            step_time = time.time()
            self.optimizer.step()
            print('optimizer step time: ', time.time() - step_time)
            if cfg.USE_EMA:
                self.ema.step(self.ema_param_dict)
        else:
            not_train_time = time.time()
            with torch.no_grad():
                batch_res = self.model.forward(batch, run_vqa, run_ref)
            print('not_train_time: ', time.time()-not_train_time)
        return batch_res
