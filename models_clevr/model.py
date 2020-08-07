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
        #groundR_time = time.time()
        proj_q = self.proj_q(vecQuestions)
        interactions = F.normalize(kb * proj_q[:, None, :], dim=-1)
        logits = self.inter2att(interactions).squeeze(-1)
        logits = ops.apply_mask1d(logits, imagesObjectNum)
        #print('GroundR time: ', time.time()-groundR_time)
        logits = torch.sigmoid(logits)
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

    def forward(self, batch, run_vqa, run_ref, train):
        forward_time = time.time()
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
        #BUILDING THE GROUND TRUTH MATRICES batch_size x num_proposals x 4
        if run_ref:
            # get gt ind, offset, and coordinates from batch 
            bboxInd = batch['bbox_ind_batch']
            bboxOffset = batch['bbox_offset_batch'] 
            bboxBatch = batch['bbox_batch'] 
            
            # initialize zero matricies for each object in each batch to store target information (these are batch_size x 20 x 4)
            bboxRefScoreGt = torch.zeros(size=(batchSize, torch.max(imagesObjectNum)), dtype=torch.float64).cuda()
            bboxOffsetGt = torch.zeros(size=(batchSize, torch.max(imagesObjectNum), 4), dtype=torch.float32).cuda()
            bboxBatchGt = torch.zeros(size=(batchSize, torch.max(imagesObjectNum), 4), dtype = torch.int64).cuda()
            
            # batch_inds is a list of the indices of the batches where there are gt positive boxes: eg: [0, 0, 0, 1, 2, 3, 3, 4, 5, 5, 5 ... 63 63] 
            # box_inds is the index of bboxInd, bboxOffset, and bboxBatch that contain a ground truth box: eg [0, 1 ,3 , 0, 2 ... 5, 0]
            # target_inds is the indices of the proposed boxes which are gt positive eg [140, 28, 32, 49, 160, ... 2]
            batch_inds = np.argwhere(bboxInd > -1)[:,0]
            box_inds = np.argwhere(bboxInd > -1)[:,1]
            target_inds = bboxInd[bboxInd > -1]
            
            bboxRefScoreGt[batch_inds[:], target_inds[:]] = 1
            bboxOffsetGt[batch_inds[:], target_inds[:], :] = torch.from_numpy(bboxOffset[batch_inds[:], box_inds[:], :].astype(np.float32)).cuda()
            bboxBatchGt[batch_inds[:], target_inds[:], :] = torch.from_numpy(bboxBatch[batch_inds[:], box_inds[:], :].astype(np.int64)).cuda()

        #print('build_gt_time: ', time.time() - build_gt_time)
        LSTM_time = time.time()

        # LSTM
        questionCntxWords, vecQuestions = self.encoder(
            questionIndices, questionLengths)
        #print('LSTM_Time: ', time.time() - LSTM_time)
        LCGN_time = time.time()
        
        # LCGN
        x_out = self.lcgn(
            images=images, q_encoding=vecQuestions,
            lstm_outputs=questionCntxWords, batch_size=batchSize,
            q_length=questionLengths, entity_num=imagesObjectNum)
        #print('LCGN_Time: ', time.time() - LCGN_time)
        
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
            if train:
                #BRYCE CODE
                print('x_out: ', x_out.shape)
                #create positive and negative sets
                #pos_set is ground truth set
                #neg_ind_k has the same number of boxes in each set as the ground truth, but they're chosen randomly
                #neg_ind_plus has the same boxes as the ground truth plus one extra random one
                #neg_ind_minus has the same boxes as the ground truth minus one (unless there was only one ground truth box)
                #do global average pooling on all of them
                num_gt_pos = torch.sum(bboxRefScoreGt, axis=1)
                num_obj = x_out.shape[1]
                feat_dim = x_out.shape[2]
                pos_set = torch.zeros(batchSize, feat_dim)
                neg_k_set = torch.zeros_like(pos_set)
                neg_plus_set = torch.zeros_like(pos_set)
                neg_minus_set = torch.zeros_like(pos_set)
                #average pooling
                for i in range(batchSize):
                    x_i = x_out[i,:,:]
                    k = int(num_gt_pos[i].item())
                    pos_ind = (bboxRefScoreGt[i,:] != 0).nonzero()
                    pos_set[i,:] = (1. / k) * torch.sum(x_out[i, pos_ind[:,0],:], dim=0)
                    #print('pos_ind: ', pos_ind.shape, ' ', pos_ind.dtype, ' ', pos_ind)
                    #print('pos_set: ', pos_set[i,:])
                    
                    neg_ind = (bboxRefScoreGt[i,:] == 0).nonzero()
                    #print('neg_ind: ', neg_ind.shape)
                    neg_ind_k = torch.randint(low=0, high=num_obj, size=(pos_ind.shape[0],1))
                    neg_k_set[i,:] = 1./ k * torch.sum(x_out[i, neg_ind_k[:,0],:], dim=0)
                    #print('neg_ind_k: ', neg_ind_k.shape, ' ', neg_ind_k)
                    #print('neg_k_set: ', neg_k_set[i,:])

                    extra_neg = torch.squeeze(neg_ind[torch.randint(low=0, high=neg_ind.shape[0], size=(1,1))], dim=0)
                    neg_ind_plus = torch.cat((pos_ind, extra_neg))
                    neg_plus_set[i,:] = 1. / (k+1) * torch.sum(x_out[i, neg_ind_plus[:,0],:], dim=0)
                    #print('neg_ind_plus: ', neg_ind_plus.shape, ' ', neg_ind_plus)
                    #print('neg_plus_set: ', neg_plus_set[i,:])

                    if pos_ind.shape[0] > 1:
                        neg_ind_minus = pos_ind[:-1].clone()
                        neg_minus_set[i,:] = 1./ (k-1) * torch.sum(x_out[i, neg_ind_minus[:,0],:], dim=0)
                        #print('set size > 1: ', neg_minus_set[i,:])
                    else:
                        neg_minus_set[i,:] = (neg_k_set[i,:] + neg_plus_set[i,:]) * .5
                        #print('set size = 1: ', neg_minus_set[i,:])
                    #print('neg_ind_minus: ', neg_ind_minus.shape, ' ', neg_ind_minus)

                #calculate ref scores for positive set
                num_neg_sets = 3
                input_sets = torch.stack((pos_set, neg_k_set, neg_plus_set, neg_minus_set), dim=1).cuda()
                #print('input_set: ', input_sets.shape)
                #print('object_num: ', imagesObjectNum.shape,' ', imagesObjectNum)
                obj_dim = (num_neg_sets+1) * torch.ones_like(imagesObjectNum)
                ref_scores = self.grounder(input_sets, vecQuestions, obj_dim)
                #print('ref_scores: ', ref_scores.shape, ' ', ref_scores)
                set_box_inds = (bboxRefScoreGt !=0).nonzero()
            
            else:
                num_beams = 3
                num_obj = x_out.shape[1]
                feat_dim = x_out.shape[2]
                # current_set_feat = the features of the current k sets (k x batchSize x feat_dim)
                current_set_feat = torch.zeros((batchSize, num_beams, feat_dim))
                #new_set_feat = the features of the sets we're exploring (k x batchSize x num_obj x feat_dim)
                new_set_feat = torch.zeros((batchSize, num_obj, feat_dim))
                #max_probs: the max probabilities for the current k sets (batchSize x k)
                k_max_probs = torch.zeros((batchSize, num_beams))
                #new_inds = the indices of the top k of the sets we're exploring (batchSize x k)
                k_new_inds = torch.zeros((batchSize, num_beams), dtype=int)
                #new_probs = the top k probabilities of the sets we're exploring (batchSize x k)
                k_new_probs = torch.zeros((batchSize, num_beams))
                #all_new_probs = the output of groundr (batchSize x num_obj)
                all_new_probs = torch.zeros((batchSize, num_obj))
                # current_box_inds = the indices of the boxes in the current k sets (k x batchSize x num_rounds-1)
                current_box_inds = torch.zeros((batchSize, num_beams, 2), dtype=int)
                #new_box_inds = the indices of the boxes in the top k sets we're exploring
                k_new_box_inds = torch.zeros((num_beams, batchSize, 2), dtype=int)
                #needed for groundr
                #obj_dim = num_beams * num_obj * torch,ones_like(imagesObjectNum)
                #unfinshed keeps track of which indices have probabilities still increasing
                unfinished = torch.arange((batchSize), dtype=int).cuda()
                finished = torch.zeros((),dtype=int).cuda()
                #num_rounds = the number of times we've searched and the number of indices in the set
                num_rounds = 1
                # probabilities: the probabilities of all the sets we're exploring (initially batchSize x 196 but will be k x batchSize x 196)
                #get the probabilities of each box
                probabilities = self.grounder(x_out, vecQuestions, imagesObjectNum)
                print('probabilities: ', probabilities.shape)
                #find the top k probabilities
                k_new_probs, k_new_inds = torch.topk(input=probabilities, k=num_beams, dim=1)
                print('k_new_probs: ', k_new_probs.shape)
                print('k_new_inds: ', k_new_inds.shape)
                #keep track of the features, probabilities, and indices of these boxes
                k_max_probs = k_new_probs
                k_new_box_inds = k_new_inds
                current_box_inds[:,:,0] = k_new_inds / batchSize
                current_box_inds[:,:,1] = k_new_inds % batchSize
                print('current_box_inds: ', current_box_inds.shape)
                current_set_feat[:,:,:] = x_out[current_box_inds[:,:,0], current_box_inds[:,:,1], :]
                print('current_set_feat: ', current_set_feat.shape)
                max_num_rounds = 30
                num_rounds = 1
                #repeat until all samples are complete:
                while len(unfinished) > 0 or num_rounds < max_num_rounds:
                    old_max_probs = k_max_probs.clone()
                    for k in range(num_beams):
                        #for each of these, combine with all other boxes
                        print('current_set_feat_k: ', current_set_feat[:,k,:].unsqueeze(1).shape)
                        expanded = (current_set_feat[:,k,:].unsqueeze(1).expand((-1,num_obj,-1)))
                        print('expanded: ', expanded.shape) 
                        print('x_out: ', x_out.is_cuda)
                        new_set_feat = (1./num_rounds) * (expanded.cuda() + x_out)
                        print('new_set_feat: ', new_set_feat.shape)
                        #calculate the probabilities of all of these sets
                        all_new_probs = self.grounder(new_set_feat, vecQuestions, imagesObjectNum)
                        print('all_new_probs: ', all_new_probs.shape)
                        #set probabilities of repeats to 0
                        all_new_probs[current_box_inds[:,k,:]] = 0
                        #find the top three probabilities for each k along the number of objects
                        k_new_probs, k_new_inds = torch.topk(input=all_new_probs, k=num_beams, dim=1)
                        print('k_new_probs: ', k_new_probs.shape)
                        print('k_new_inds: ', k_new_inds.shape)
                        #compare with max_probs:
                        #if the probability has increased, update current box_inds, current_set_feat, and new_max_probs
                        increase_inds = ((k_new_probs - k_max_probs)>0).nonzero()
                        print('increase_inds: ', increase_inds.shape)
                        k_max_probs[increase_inds] = k_new_probs[increase_inds]
                        print('k_max_probs: ', k_max_probs.shape)
                        #current_box_inds needs to include the old indices as well, so this needs to be appended rather than replaced 
                        #CHECK THIS
                        #new_box_inds should be the indices for the boxes related to max_probs
                        k_new_box_inds[increase_inds] = k_new_inds[increase_inds]
                        print('k_new_box_inds: ', k_new_box_inds.shape)
                        # THIS NEEDS TO BE FIXED!!!
                        current_set_feat[:,:,:] = new_set_feat[(k_new_box_inds / batchSize), (k_new_box_inds % batchSize),:]
                        print('current_set_feat: ', current_set_feat.shape)
                    #for each sample, if the probability has decreased, save the indices of the boxes in the current set to the output
                    max_1_old = torch.max(old_max_probs, dim=1)[0]
                    max_1_new = torch.max(k_max_probs, dim=1)[0]
                    print('max_1_old: ', max_1_old.shape, ' max_1_new: ', max_1_new.shape)
                    difference_in_max_1 = max_1_old - max_1_new
                    print('difference_in_max_1: ', difference_in_max_1.shape)
                    #decrease_inds is the indices of the samples where the maximum probability has decreased.  Increase_inds is where it has increased.  The lengths of these sum to batchSize
                    decrease_inds = torch.squeeze((difference_in_max_1 > 0).nonzero())
                    increase_inds = torch.squeeze((difference_in_max_1 <= 0).nonzero())
                    print('decrease_inds: ', decrease_inds.shape, ' ', decrease_inds.is_cuda, ' increase_inds: ', increase_inds.shape, ' ', increase_inds.is_cuda)
                    #unfinished decrease_inds should be the intersections of decrease_inds and unfinished 
                    unfinished_decrease_inds = torch.from_numpy(np.intersect1d(decrease_inds.detach().cpu().numpy(), unfinished.detach().cpu().numpy())).cuda()
                    unfinished_increase_inds = torch.from_numpy(np.intersect1d(increase_inds.detach().cpu().numpy(), unfinished.detach().cpu().numpy())).cuda()
                    print('unfinished_decrease_inds: ', unfinished_decrease_inds.shape, ' unfinished_increase_inds: ', unfinished_increase_inds.shape)
                    output_box_inds = torch.append(output_box_inds, current_box_inds[unfinished_decrease_inds])
                    print('output_box_inds: ', output_box_inds.shape)
                    #keep track of which samples are complete
                    finished = torch.append(finished, unfinished_decrease_inds)
                    unfinished = unfinished.remove(unfinished_decrease_inds)
                    print('finished: ', len(finished), ' ', finished)
                    print('unfinished: ', len(unfinished), ' ', unfinished)
                    print('total: ', len(finished) + len(unfinished))
                    #for those where the probability increased, save the new probabilites, indices, and features and repeat
                    #we want append_boxes to be num_total_boxes x 3
                    append_boxes = k_new_box_inds[unfinished_increase_inds, :, :]
                    print('append_boxes: ', append_boxes.shape)
                    current_box_inds.append(k_new_box_inds[unfinished_increase_inds])
                    pause
                    num_rounds += 1
                #set box_inds is equal to the output box indices
                set_box_inds = output_box_inds
            
            # calculate bbox_offset (this was not trained)
            bbox_offset, bbox_offset_fcn = self.bbox_regression(x_out, set_box_inds)
            
            # bbox predictions returns a matrix that is batch_size x num_boxes x 4.  
            # it has the predicted x,y,w,h of all bounding boxes with matching scores higher than the threshold, all other coordinates are 0 
            bbox_predictions = batch_feat_grid2bbox(set_box_inds.detach().cpu().numpy(), bboxBatchGt.shape,bbox_offset.detach().cpu().numpy(),cfg.IMG_H / cfg.H_FEAT, cfg.IMG_W / cfg.W_FEAT,cfg.H_FEAT, cfg.W_FEAT)
            
            #calculate the loss
            loss_time = time.time()
            set_loss, bbox_offset_loss = self.add_bbox_loss_op(ref_scores, bbox_offset_fcn, bboxRefScoreGt, bboxOffsetGt)
            loss += (set_loss + bbox_offset_loss)
            
            # for normal version, calculate box ious to use as a metric
            bbox_ious = batch_bbox_iou(bbox_predictions, bboxBatchGt, bboxRefScoreGt)
            bbox_num_correct = np.sum(bbox_ious >= cfg.BBOX_IOU_THRESH)
            
            # calculate number of positives, negatives, and auc using function
            #true_positive, total_positive, true_negative, total_negative, precision, top_accuracy_list, auc, f1 = self.calc_correct(bboxrefscoregt, ref_scores)
            
            res_update_time = time.time()
            possible_correct = float(bboxRefScoreGt.shape[0]*bboxRefScoreGt.shape[1])
            possible_correct_boxes = torch.sum(bboxRefScoreGt).item()
            print('one batch!')
            res.update({
                #"accuracy_list" : top_accuracy_list,
                "bbox_predictions": bbox_predictions,
                "gt_coords": bboxBatchGt,
                "bbox_ious": bbox_ious,
                #"true_positive": int(true_positive),
                #"true_negative": int(true_negative),
                #"false_positive": int(total_negative-true_negative),
                #"false_negative": int(total_positive - true_positive),
                "bbox_num_correct": int(bbox_num_correct),
                #"num_correct": int(true_positive + true_negative),
                #"top_accuracy": float(np.mean(top_accuracy_list)),
                "bbox_accuracy": float((bbox_num_correct * 1.)/possible_correct_boxes),
                #"possible_correct": float(possible_correct),
                "possible_correct_boxes": int(possible_correct_boxes),
                #"precision": float(precision),
                #"pr_auc": float(auc),
                #"pr_f1": f1
            #bryce code
            })
        res.update({"batch_size": int(batchSize), "loss": loss})
        print('forward time: ', time.time() - forward_time)
        return res

    #bryce code
    def calc_correct(self, gt_scores, ref_scores):
        calc_correct_time = time.time()
        
        # slice inds is the indices where the ground truth positives are
        slice_inds = (gt_scores !=0).nonzero()
        total_positive = slice_inds.shape[0]
        #print('total_positive: ', total_positive)
        # slice_inds_neg is the indices where the ground truth negatives are
        slice_inds_neg = (gt_scores == 0).nonzero()
        total_negative = slice_inds_neg.shape[0]
        #print('total_negative: ', total_negative)
        #the means of the values for the probabilities at gt positive and gt negative indiceis
        
        batch_size = ref_scores.shape[0]
        probabilities = ref_scores.clone().detach().cpu().numpy()
        gt = gt_scores.detach().cpu().numpy()
       
        #calculate acu
        batch_size = ref_scores.shape[0]
        if batch_size == 1:
            gt = np.expand_dims(gt, axis=0)
            probabilities = np.expand_dims(probabilities, axis=0)
        auc = 0
        f1 = 0
        for b in range(batch_size):
            pr_precision = dict() 
            pr_recall = dict()
            pr_auc = dict()
            pr_f1 = dict()
            pr_precision[b], pr_recall[b], _ = precision_recall_curve(gt[b,:].t, probabilities[b,:].t)
            pr_auc[b] = auc(pr_recall[b], pr_precision[b])
            pr_f1[b] = f1_score(gt[b,:].t, thresh_classifications[b,:].t)
            #print('f1: ', pr_f1)
            #print('auc: ', pr_auc)
            auc += pr_auc[b]
            f1 += pr_f1[b]
            #print('\nauc is ', roc_auc[b])
        auc = auc / batch_size
        f1 = f1 / batch_size
        
        # recalculate for thresh in config = 0.9 and return results
        print('\n\n threshold: ', cfg.match_thresh)
        print('precisions: ', precision)
        print('recall: ', recall)
        print('true positive: ', true_positive, ' false positive: ', total_negative - true_negative)
        print('correct: ', true_negative + true_positive, ' incorrect: ', gt_scores.shape[0]*gt_scores.shape[1]-(true_negative + true_positive))
        print('accuracy: ', np.mean(top_accuracy))
        print('average pr auc: ', auc)
        print('average pr f1: ', f1)
        #print('calc_correct_time: ', time.time()-calc_correct_time)
        return (true_positive, total_positive, true_negative, total_negative, precision, top_accuracy, auc, f1)

    def add_pred_op(self, logits, answers):
        if cfg.mask_padunk_in_logits:
            logits = logits.clone()
            logits[..., :2] += -1e30  # mask <pad> and <unk>

        preds = torch.argmax(logits, dim=-1).detach()
        corrects = (preds == answers)
        correctnum = torch.sum(corrects).item()
        preds = preds.cpu().numpy()

        return preds, correctnum

    def add_answer_loss_op(self, logits, answers):
        if cfg.train.loss_type == "softmax":
            loss = f.cross_entropy(logits, answers)
        elif cfg.train.loss_type == "sigmoid":
            answerDist = F.one_hot(answers, self.num_choices).float()
            loss = F.binary_cross_entropy_with_logits(
                logits, answerDist) * self.num_choices
        else:
            raise Exception("non-identified loss")
        return loss

    #DEBUG
    def add_bbox_loss_op(self, ref_scores, bbox_offset_fcn, bbox_ind_gt, bbox_offset_gt):
        #BRYCE CODE
        # bounding box selection loss
        pos_prob = ref_scores[:,0].unsqueeze(1)
        #print('pos_prob: ', pos_prob.shape, ' ', pos_prob)
        neg_prob = ref_scores[:,1:]
        #print('neg_prob: ', neg_prob.shape, ' ', neg_prob)
        set_loss = torch.relu(neg_prob.unsqueeze(1) - pos_prob.unsqueeze(-1) + cfg.MARGIN).mean()
       
        #print('pos_prob_avg: ', torch.mean(pos_prob).item(), '\nk-random_avg: ', torch.mean(neg_prob[:,0]).item(), '\nk+1_avg: ', torch.mean(neg_prob[:,1]).item(), '\nk-1_avg: ', torch.mean(neg_prob[:,2]).item())
        #print('set_loss: ', set_loss)

        slice_inds = (bbox_ind_gt != 0).nonzero()
        bbox_ind_gt_sliced = bbox_ind_gt[slice_inds[:,0], slice_inds[:,1]]
        #print('slice_inds: ', slice_inds.shape)
        
        bbox_offset_sliced = bbox_offset_fcn[slice_inds[:,0], slice_inds[:,1], :]
        gt_offset_sliced = bbox_offset_gt[slice_inds[:,0], slice_inds[:,1], :]
        
        bbox_offset_loss = F.mse_loss(bbox_offset_sliced, gt_offset_sliced)
        #BRYCE CODE
        return set_loss, bbox_offset_loss


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
            #print('start_time: ', time.time()-start_time)
            forward_time = time.time()
            batch_res = self.model.forward(batch, run_vqa, run_ref, train)
            #print('forward_time: ', time.time() - forward_time)
            backward_time = time.time()
            loss = batch_res['loss']
            loss.backward()
            #print('backward_time: ', time.time() - backward_time)
            if cfg.TRAIN.CLIP_GRADIENTS:
                clip_grad_time = time.time()
                nn.utils.clip_grad_norm_(
                    self.trainable_params, cfg.TRAIN.GRAD_MAX_NORM)
                #print('clip_grad_time: ', time.time()-clip_grad_time)
            step_time = time.time()
            self.optimizer.step()
            #print('optimizer step time: ', time.time() - step_time)
            if cfg.USE_EMA:
                self.ema.step(self.ema_param_dict)
        else:
            not_train_time = time.time()
            with torch.no_grad():
                batch_res = self.model.forward(batch, run_vqa, run_ref, train)
            #print('not_train_time: ', time.time()-not_train_time)
        return batch_res
