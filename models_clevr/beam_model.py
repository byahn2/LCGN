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
                #create positive and negative sets
                #pos_set is ground truth set
                #neg_ind_k has the same number of boxes in each set as the ground truth, but they're chosen randomly
                #neg_ind_plus has the same boxes as the ground truth plus one extra random one
                #neg_ind_minus has the same boxes as the ground truth minus one (unless there was only one ground truth box)
                #do global average pooling on all of them
                num_gt_pos = torch.sum(bboxRefScoreGt, axis=1)
                num_obj = x_out.shape[1]
                feat_dim = x_out.shape[2]
                pos_mask = torch.zeros(batchSize, num_obj).cuda()
                neg_k_mask = torch.zeros_like(pos_mask).cuda()
                neg_plus_mask = torch.zeros_like(pos_mask).cuda()
                neg_minus_mask = torch.zeros_like(pos_mask).cuda()

                #average pooling
                for i in range(batchSize):
                    k = int(num_gt_pos[i].item())
                    pos_ind = (bboxRefScoreGt[i,:] != 0).nonzero()
                    pos_mask[i, pos_ind] = 1/k
                    #print('pos_ind: ', pos_ind.shape, ' ', pos_ind.dtype, ' ', pos_ind)
                    
                    neg_ind = (bboxRefScoreGt[i,:] == 0).nonzero()
                    #print('neg_ind: ', neg_ind.shape)
                    neg_ind_k = torch.randint(low=0, high=num_obj, size=(k,1))
                    neg_k_mask[i, neg_ind_k] = 1/k
                    #print('neg_ind_k: ', neg_ind_k.shape, ' ', neg_ind_k)

                    extra_neg = torch.squeeze(neg_ind[torch.randint(low=0, high=neg_ind.shape[0], size=(1,1))], dim=0)
                    neg_ind_plus = torch.cat((pos_ind, extra_neg))
                    neg_plus_mask[i, neg_ind_plus] = 1/(k+1)
                    #print('neg_ind_plus: ', neg_ind_plus.shape, ' ', neg_ind_plus)
                    neg_ind_minus = pos_ind[:-1].clone()
                    if k > 1:
                        neg_minus_mask[i, neg_ind_minus] = 1/(k-1)
                    else:
                        neg_ind_minus = torch.randint(low=0, high=num_obj, size=(k,1))
                        neg_minus_mask[i, neg_ind_minus] = 1/k
                    #print('neg_ind_minus: ', neg_ind_minus.shape, ' ', neg_ind_minus)

                #calculate ref scores for positive set
                num_neg_sets = 3
                pos_set = torch.sum((pos_mask.unsqueeze(-1) * x_out),dim=1)
                #print('pos_mask: ', pos_mask[0,:])
                neg_k_set = torch.sum((neg_k_mask.unsqueeze(-1) * x_out),dim=1)
                neg_plus_set = torch.sum((neg_plus_mask.unsqueeze(-1) * x_out), dim=1)
                neg_minus_set = torch.sum((neg_plus_mask.unsqueeze(-1) * x_out), dim=1)
                input_sets = torch.stack((pos_set, neg_k_set, neg_plus_set, neg_minus_set), dim=1).cuda()
                #print('object_num: ', imagesObjectNum.shape,' ', imagesObjectNum)
                obj_dim = (num_neg_sets+1) * torch.ones_like(imagesObjectNum)
                ref_scores = self.grounder(input_sets, vecQuestions, obj_dim)
                set_box_inds = (bboxRefScoreGt !=0).nonzero()
                print('training_set')
            else:
                num_beams = 1
                num_obj = x_out.shape[1]
                feat_dim = x_out.shape[2]
                #num_rounds = the number of times we've searched and the number of indices in the set
                num_rounds = 1
                max_num_rounds = 1
                # current_set_feat = the features of the current k sets (batchsize x k x feat_dim)
                current_set_feat = torch.zeros((batchSize, num_beams, feat_dim)).cuda()
                # current_box_inds = the indices of the boxes in the current k sets (batchsize x k x num_rounds-1)
                current_box_inds = torch.zeros((batchSize, num_beams, 2), dtype=int).cuda()
                #output_box_inds = 
                output_box_inds = torch.zeros((0, 2), dtype=int).cuda()
                #unfinshed keeps track of which indices have probabilities still increasing
                unfinished = torch.arange((batchSize), dtype=int)
                finished = torch.zeros((),dtype=int).cuda()
                unfinished_increase_inds = torch.zeros((0,1),dtype=int).cuda()
                unfinished_decrease_inds = torch.zeros((0,1), dtype=int).cuda()
                # probabilities: the probabilities of all the sets we're exploring (initially batchsize x 196 but will be k x batchsize x 196)
                #get the probabilities of each box
                #all_new_probs = the output of groundr (batchsize x num_obj)
                all_new_probs = self.grounder(x_out, vecQuestions, imagesObjectNum)
                #print('probabilities: ', all_new_probs.shape)
                #find the top k probabilities
                #k_new_probs = the top k probabilities of the sets we're exploring (batchsize x k)
                #k_new_inds = the indices of the top k of the sets we're exploring (batchsize x k)
                k_new_probs, k_new_inds = torch.topk(input=all_new_probs, k=num_beams, dim=1)
                #print('k_new_probs: ', k_new_probs.shape)
                #print('k_new_inds: ', k_new_inds.shape)
                #keep track of the features, probabilities, and indices of these boxes
                #k_max_probs: the max probabilities for the current k sets (batchsize x k)
                k_max_probs = k_new_probs.clone()
                #k_new_box_inds = the indices of the boxes in the top k sets we're exploring
                k_new_box_inds = k_new_inds.clone()
                current_box_inds[:,:,0] = torch.arange(batchSize, dtype=int).unsqueeze(dim=1).expand((-1, num_beams))
                current_box_inds[:,:,1] = k_new_inds
                #print('current_box_inds: ', current_box_inds.shape)
                current_set_feat[:,:,:] = x_out[current_box_inds[:,:,0], current_box_inds[:,:,1], :]
                #print('current_set_feat: ', current_set_feat.shape)
                #repeat until all samples are complete:
                while len(unfinished) > 0 and num_rounds < max_num_rounds:
                    old_max_probs = k_max_probs.clone()
                    for k in range(num_beams):
                        #for each of these, combine with all other boxes
                        #print('current_set_feat_k: ', current_set_feat[:,k,:].unsqueeze(1).shape)
                        #new_set_feat = the features of the sets we're exploring ( batchsize x num_obj x feat_dim)
                        new_set_feat = (1./num_rounds) * ((current_set_feat[:,k,:].unsqueeze(dim=1).expand((-1,num_obj,-1)))
 + x_out)
                        #print('new_set_feat: ', new_set_feat.shape)
                        #calculate the probabilities of all of these sets
                        k_new_probs, k_new_inds = torch.topk(input=all_new_probs, k=num_beams, dim=1)
                        all_new_probs = self.grounder(new_set_feat, vecQuestions, imagesObjectNum)
                        #print('all_new_probs: ', all_new_probs.shape)
                        k_new_probs, k_new_inds = torch.topk(input=all_new_probs, k=num_beams, dim=1)
                        #set probabilities of repeats to 0
                        all_new_probs[current_box_inds[:,k,0], current_box_inds[:,k,1]] = 0
                        #find the top three probabilities for each k along the number of objects
                        k_new_probs, k_new_inds = torch.topk(input=all_new_probs, k=num_beams, dim=1)
                        #print('k_new_probs: ', k_new_probs.shape)
                        #print('k_new_inds: ', k_new_inds.shape)
                        #compare with max_probs:
                        #if the probability has increased, update current box_inds, current_set_feat, and new_max_probs
                        increase_inds = ((k_new_probs - k_max_probs)>0).nonzero()
                        #print('increase_inds: ', increase_inds.shape)
                        if increase_inds.shape[0] > 0:
                            k_max_probs[increase_inds] = k_new_probs[increase_inds]
                            #print('k_max_probs: ', k_max_probs.shape)
                            #current_box_inds needs to include the old indices as well, so this needs to be appended rather than replaced 
                            #new_box_inds should be the indices for the boxes related to max_probs
                            k_new_box_inds[increase_inds] = k_new_inds[increase_inds]
                            #print('k_new_box_inds: ', k_new_box_inds.shape)

                            #anywhere where the probability increased, change the previous boxes in the set to the boxes in the increase set 
                            #print('current_box_increase_inds: ', current_box_inds[:, increase_inds, :].shape)
                            #print('current_box_k_inds: ', current_box_inds[:,k,:].shape)
                            current_box_inds[:, increase_inds, :] = (current_box_inds[:, k, :]).unsqueeze(dim=1).expand(-1,increase_inds.shape[0],-1)
                            #print('current_box_inds: ', current_box_inds.shape)
                            #print('current_box_inds: ', current_box_inds)

                            # this needs to be fixed!!!
                            current_set_feat[:,:,:] = new_set_feat[torch.arange(batchSize, dtype=int).unsqueeze(dim=1).expand((-1,num_beams)), k_new_box_inds,:]
                            #print('current_set_feat: ', current_set_feat.shape)
                            #print('current_set_feat: ', current_set_feat[0,0,:])
                    #for each sample, if the probability has decreased, save the indices of the boxes in the current set to the output
                    difference_in_max_1 = torch.max(old_max_probs, dim=1)[0] - torch.max(k_max_probs, dim=1)[0]
                    #print('difference_in_max_1: ', difference_in_max_1.shape)
                    #decrease_inds is the indices of the samples where the maximum probability has decreased.  increase_inds is where it has increased.  the lengths of these sum to batchsize
                    decrease_inds = torch.squeeze((difference_in_max_1 > 0).nonzero())
                    increase_inds = torch.squeeze((difference_in_max_1 <= 0).nonzero())
                    #print('decrease_inds: ', decrease_inds.shape, ' ', decrease_inds.is_cuda, ' increase_inds: ', increase_inds.shape, ' ', increase_inds.is_cuda)
                    #unfinished decrease_inds should be the intersections of decrease_inds and unfinished 
                    unfinished_decrease_inds = torch.from_numpy(np.intersect1d(decrease_inds.detach().cpu().numpy(), unfinished.detach().cpu().numpy())).cuda()
                    unfinished_increase_inds = torch.from_numpy(np.intersect1d(increase_inds.detach().cpu().numpy(), unfinished.detach().cpu().numpy())).cuda()
                    #print('unfinished_decrease_inds: ', unfinished_decrease_inds.shape, ' unfinished_increase_inds: ', unfinished_increase_inds.shape)
                    max_1_inds = torch.max(k_max_probs, dim=1)[1]
                    if output_box_inds.shape[0] == 0:
                        output_box_inds = current_box_inds[unfinished_decrease_inds, max_1_inds[unfinished_decrease_inds],:]
                        finished = unfinished_decrease_inds
                    else:
                        output_box_inds = torch.cat((output_box_inds, current_box_inds[unfinished_decrease_inds,max_1_inds[unfinished_decrease_inds],:]), dim=0)
                        finished = torch.cat((finished, unfinished_decrease_inds))
                    #print('output_box_inds: ', output_box_inds.shape)
                    #print('finished: ', len(finished), ' ', finished)
                    #keep track of which samples are complete
                    for e in unfinished_decrease_inds:
                        if (unfinished == e).any():
                            index_not_e = torch.squeeze((unfinished != e).nonzero())
                            unfinished = unfinished[index_not_e]
                    #for those where the probability increased, save the new probabilites, indices, and features and repeat
                    # update the max_inds where the probability increased
                    append_boxes = torch.empty((len(unfinished_increase_inds), num_beams,2), dtype=int).cuda()
                    append_boxes[:,:,0] = unfinished_increase_inds.unsqueeze(dim=1).expand((-1,num_beams))
                    append_boxes[:,:,1] = k_new_box_inds[unfinished_increase_inds, :]
                    #print('append_boxes: ', append_boxes.shape)
                    current_box_inds = torch.cat((current_box_inds, append_boxes), dim=0)
                    #print('current_box_inds: ', current_box_inds.shape)
                    num_rounds += 1
                    print('')
                #set box_inds is equal to the output box indices
                print('unfinished: ', len(unfinished), ' round_num: ', num_rounds)
                if len(unfinished >=0):
                    max_1_probabilities, max_1_inds = torch.max(k_max_probs, dim=1)
                    #print('to_append: ', current_box_inds[unfinished_increase_inds, max_1_inds[unfinished_increase_inds],:].shape)
                    if max_num_rounds > 1 and num_beams > 1:
                        output_box_inds = torch.cat((output_box_inds, current_box_inds[unfinished_increase_inds, max_1_inds[unfinished_increase_inds], :]), dim=0)
                    else:
                        output_box_inds = torch.squeeze(current_box_inds)
                        #print('output_box_inds: ', output_box_inds.shape)
                set_box_inds = output_box_inds
                print('eval')
            
            print('set_box_inds: ', set_box_inds.shape)
            # calculate bbox_offset (this was not trained)
            bbox_offset, bbox_offset_fcn = self.bbox_regression(x_out, set_box_inds)
            
            # bbox predictions returns a matrix that is batch_size x num_boxes x 4.  
            # it has the predicted x,y,w,h of all bounding boxes with matching scores higher than the threshold, all other coordinates are 0 
            bbox_predictions = batch_feat_grid2bbox(set_box_inds.detach().cpu().numpy(), bboxBatchGt.shape,bbox_offset.detach().cpu().numpy(),cfg.IMG_H / cfg.H_FEAT, cfg.IMG_W / cfg.W_FEAT,cfg.H_FEAT, cfg.W_FEAT)
            
            
            #calculate the loss
            loss_time = time.time()
            bbox_offset_loss = self.add_bbox_loss_op(bbox_offset_fcn, bboxRefScoreGt, bboxOffsetGt)
            if train:
                set_loss = self.add_set_loss_op(ref_scores)
                loss += (set_loss + bbox_offset_loss)
            else:
                loss += bbox_offset_loss
            
            # for normal version, calculate box ious to use as a metric
            bbox_ious = batch_bbox_iou(bbox_predictions, bboxBatchGt, bboxRefScoreGt)
            bbox_num_correct = np.sum(bbox_ious >= cfg.BBOX_IOU_THRESH)
            print('bbox_num_correct: ', bbox_num_correct)
            
            # calculate number of positives, negatives, and auc using function
            true_positive, total_positive, true_negative, total_negative, precision, top_accuracy_list = self.calc_correct(bboxRefScoreGt, set_box_inds)
            print('top_accuracy: ', np.mean(top_accuracy_list))   
            res_update_time = time.time()
            possible_correct = float(bboxRefScoreGt.shape[0]*bboxRefScoreGt.shape[1])
            possible_correct_boxes = torch.sum(bboxRefScoreGt).item()
            print('one batch!')
            res.update({
                "accuracy_list" : top_accuracy_list,
                "bbox_predictions": bbox_predictions,
                "gt_coords": bboxBatchGt,
                "bbox_ious": bbox_ious,
                "true_positive": int(true_positive),
                "true_negative": int(true_negative),
                "false_positive": int(total_negative-true_negative),
                "false_negative": int(total_positive - true_positive),
                "bbox_num_correct": int(bbox_num_correct),
                "num_correct": int(true_positive + true_negative),
                "top_accuracy": float(np.mean(top_accuracy_list)),
                "bbox_accuracy": float((bbox_num_correct * 1.)/possible_correct_boxes),
                "possible_correct": float(possible_correct),
                "possible_correct_boxes": int(possible_correct_boxes),
                "precision": float(precision),
                #"pr_auc": float(auc),
                #"pr_f1": f1
            #bryce code
            })
        res.update({"batch_size": int(batchSize), "loss": loss})
        print('forward time: ', time.time() - forward_time)
        return res

    #bryce code
    def calc_correct(self, gt_scores, set_box_inds):
        calc_correct_time = time.time()
        
        classifications = torch.zeros(gt_scores.shape)
        classifications[set_box_inds[:,0], set_box_inds[:,1]] = 1
        batch_size = gt_scores.shape[0]

        # slice inds is the indices where the ground truth positives are
        slice_inds = (gt_scores !=0).nonzero()
        total_positive = slice_inds.shape[0]
        gt_pos_slice = classifications[slice_inds[:,0], slice_inds[:,1]]
        #print('total_positive: ', total_positive)
        # slice_inds_neg is the indices where the ground truth negatives are
        slice_inds_neg = (gt_scores == 0).nonzero()
        total_negative = slice_inds_neg.shape[0]
        gt_neg_slice = classifications[slice_inds_neg[:,0], slice_inds_neg[:,1]]
        #print('total_negative: ', total_negative)
        #the means of the values for the probabilities at gt positive and gt negative indiceis
       
        gt = gt_scores.detach().cpu().numpy()
        np_class = classifications.detach().cpu().numpy()
        #caluclate top accuracy
        num_gt_pos = np.sum(gt, axis=1)
        #print('num_gt_pos:' , num_gt_pos.shape)
        sorted_probs = np.flip(np.sort(np_class, axis=1), axis=1)
        #print('sorted_probs: ', sorted_probs.shape)
        top_pos = np.zeros(num_gt_pos.shape, dtype=float)
        for i in range(len(num_gt_pos)):
            n = int(num_gt_pos[i])
            top_pos[i] = sum(sorted_probs[i, 0:n])
        #print('top_pos: ', top_pos.shape)
        top_accuracy = top_pos / num_gt_pos
        


        #for checking k=1 max_rounds = 1:
        #slice_inds_class = (classifications == 1).nonzero()
        #top_accuracy = gt_scores[slice_inds_class[:,0], slice_inds_class[:,1]].detach().cpu().numpy()
        #print('top_accuracy_for k=1 and max_rounds = 1: ', np.mean(top_accuracy))
        #print('top_accuracy: ', top_accuracy.shape, ' ', top_accuracy)

        #calculate acu
        #if batch_size == 1:
        #    gt = np.expand_dims(gt, axis=0)
        #    np_class = np.expand_dims(np_class, axis=0)
        #auc = 0
        #f1 = 0
        #for b in range(batch_size):
        #    print('gt[b]: ', gt[b,:].t)
        #    print('np[b]: ', np_class[b,:].t)
        #    pr_precision = dict() 
        #    pr_recall = dict()
        #    pr_auc = dict()
        #    pr_f1 = dict()
        #    pr_precision[b], pr_recall[b], _ = precision_recall_curve(gt[b,:].t, np_class[b,:].t)
        #    pr_auc[b] = auc(pr_recall[b], pr_precision[b])
        #    pr_f1[b] = f1_score(gt[b,:].t, np_class[b,:].t)
        #    #print('f1: ', pr_f1)
        #    #print('auc: ', pr_auc)
        #    auc += pr_auc[b]
        #    f1 += pr_f1[b]
        #auc = auc / batch_size
        #f1 = f1 / batch_size


        true_positive = len(gt_pos_slice[gt_pos_slice == 1])
        true_negative = len(gt_neg_slice[gt_neg_slice == 0]) 
        false_negative = total_positive - true_positive
        false_positive = total_negative - true_negative
        print('true_positive: ', true_positive, ' total_positive: ', total_positive, ' false_positive: ', false_positive)
        print('true_negative: ', true_negative, ' total negative: ', total_negative, ' false_negative: ', false_negative)
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)

        
        # recalculate for thresh in config = 0.9 and return results
        print('precisions: ', precision)
        print('recall: ', recall)
        print('true positive: ', true_positive, ' false positive: ', total_negative - true_negative)
        print('correct: ', true_negative + true_positive, ' incorrect: ', gt_scores.shape[0]*gt_scores.shape[1]-(true_negative + true_positive))
        print('accuracy: ', np.mean(top_accuracy))
        #print('average pr auc: ', auc)
        #print('average pr f1: ', f1)
        #print('calc_correct_time: ', time.time()-calc_correct_time)
        return (true_positive, total_positive, true_negative, total_negative, precision, top_accuracy)#, auc, f1)

    def add_pred_op(self, logits, answers):
        if cfg.MASK_PADUNK_IN_LOGITS:
            logits = logits.clone()
            logits[..., :2] += -1e30  # mask <pad> and <unk>

        preds = torch.argmax(logits, dim=-1).detach()
        corrects = (preds == answers)
        correctnum = torch.sum(corrects).item()
        preds = preds.cpu().numpy()

        return preds, correctnum

    def add_answer_loss_op(self, logits, answers):
        if cfg.TRAIN.LOSS_TYPE == "softmax":
            loss = F.cross_entropy(logits, answers)
        elif cfg.TRAIN.LOSS_TYPE == "sigmoid":
            answerdist = F.one_hot(answers, self.num_choices).float()
            loss = F.binary_cross_entropy_with_logits(
                logits, answerdist) * self.num_choices
        else:
            raise Exception("non-identified loss")
        return loss

    #debug
    def add_bbox_loss_op(self, bbox_offset_fcn, bbox_ind_gt, bbox_offset_gt):
        #bryce code
        slice_inds = (bbox_ind_gt != 0).nonzero()
        bbox_ind_gt_sliced = bbox_ind_gt[slice_inds[:,0], slice_inds[:,1]]
        #print('slice_inds: ', slice_inds.shape)
        
        bbox_offset_sliced = bbox_offset_fcn[slice_inds[:,0], slice_inds[:,1], :]
        gt_offset_sliced = bbox_offset_gt[slice_inds[:,0], slice_inds[:,1], :]
        
        bbox_offset_loss = F.mse_loss(bbox_offset_sliced, gt_offset_sliced)
        #bryce code
        return bbox_offset_loss
    
    def add_set_loss_op(self, ref_scores):
        # bounding box selection loss
        pos_prob = ref_scores[:,0].unsqueeze(1)
        #print('pos_prob: ', pos_prob.shape, ' ', pos_prob)
        neg_prob = ref_scores[:,1:]
        #print('neg_prob: ', neg_prob.shape, ' ', neg_prob)
        set_loss = torch.relu(neg_prob.unsqueeze(1) - pos_prob.unsqueeze(-1) + cfg.MARGIN).mean() 
        print('pos_prob_avg: ', torch.mean(pos_prob).item(), '\nk-random_avg: ', torch.mean(neg_prob[:,0]).item(), '\nk+1_avg: ', torch.mean(neg_prob[:,1]).item(), '\nk-1_avg: ', torch.mean(neg_prob[:,2]).item())
        #print('set_loss: ', set_loss)
        return set_loss

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
