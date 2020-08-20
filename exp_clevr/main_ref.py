import os
import numpy as np
import json
import torch
import time

from models_clevr.model import LCGNwrapper
from models_clevr.config import build_cfg_from_argparse
from util.clevr_train.data_reader import DataReader

# Load config
cfg = build_cfg_from_argparse()

# Start session
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPUS
if len(cfg.GPUS.split(',')) > 1:
    print('PyTorch implementation currently only supports single GPU')


def load_train_data(max_num=0):
    load_train_time = time.time()
    imdb_file = cfg.IMDB_FILE % cfg.TRAIN.SPLIT_REF
    data_reader = DataReader(
        imdb_file, shuffle=True, max_num=max_num,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        vocab_question_file=cfg.VOCAB_QUESTION_FILE,
        T_encoder=cfg.T_ENCODER,
        vocab_answer_file=cfg.VOCAB_ANSWER_FILE,
        load_spatial_feature=True,
        spatial_feature_dir=cfg.SPATIAL_FEATURE_DIR,
        add_pos_enc=cfg.ADD_POS_ENC, img_H=cfg.IMG_H, img_W=cfg.IMG_W,
        pos_enc_dim=cfg.PE_DIM, pos_enc_scale=cfg.PE_SCALE)
    #print('after data reader')
    num_vocab = data_reader.batch_loader.vocab_dict.num_vocab
    num_choices = data_reader.batch_loader.answer_dict.num_vocab
    print('load_train_time: ', time.time()-load_train_time)
    return data_reader, num_vocab, num_choices


def run_train_on_data(model, data_reader_train, lr_start,
                      run_eval=False, data_reader_eval=None):
    model.train()
    lr = lr_start
    correct, total, loss_sum, batch_num, top_acc = 0, 0, 0., 0, 0.
    prev_loss = None
    for batch, n_sample, e in data_reader_train.batches(one_pass=False):
        batch_time = time.time()
        n_epoch = cfg.TRAIN.START_EPOCH + e
        if n_sample == 0 and n_epoch > cfg.TRAIN.START_EPOCH:
            snapshot_time = time.time()
            ##save snapshot
            snapshot_file = cfg.SNAPSHOT_FILE % (cfg.EXP_NAME, n_epoch)
            save_model_time = time.time()
            torch.save(model.state_dict(), snapshot_file)
            #print('\nsave_model_time: ', time.time() - save_model_time)
            states_file = snapshot_file.replace('.ckpk', '') + '_states.npy'
            np.save(states_file, {'lr': lr})
            #print('snapshot_time: ', time.time() - snapshot_time)
            eval_time = time.time()
            # run evaluation
            if run_eval:
                run_eval_on_data(model, data_reader_eval)
                model.train()
            #print('eval_time: ', time.time() - eval_time)
            adjust_time = time.time()
            # adjust lr:
            curr_loss = loss_sum/batch_num
            if prev_loss is not None:
                lr = adjust_lr_clevr(curr_loss, prev_loss, lr)
            #print('adjust_lr_time: ', time.time() - adjust_time)
            clear_stats_time = time.time()
            # clear stats
            correct, total, loss_sum, batch_num, top_acc = 0, 0, 0., 0, 0.
            prev_loss = curr_loss
            #print('clear_stats_time: ', time.time() - clear_stats_time)
        
        batch_res_time = time.time()
        if n_epoch >= cfg.TRAIN.MAX_EPOCH:
            break
        batch_res = model.run_batch(
            batch, train=True, run_vqa=False, run_ref=True, lr=lr)
        #print('batch_res_time: ', time.time()-batch_res_time)
        #BRYCE CODE

        record_time = time.time()
        correct += batch_res['bbox_num_correct']
        total += batch_res['possible_correct_boxes']
        top_acc += batch_res['top_accuracy']
        #print('correct: ', correct, ' total: ', total, ' accuracy: ', correct/total)
        #BRYCE CODE
        loss_sum += batch_res['loss'].item()
        batch_num += 1
        print('\rTrain E %d S %d: avgL=%.4f, avgboxA=%.4f, avgtopA=%.4f, lr=%.1e' % (n_epoch+1, total, loss_sum/batch_num, correct/total, top_acc/batch_num, lr), end='')
        #print('record_time: ', time.time()-record_time)
        print('\n1 batch: ', time.time() - batch_time)
        #BRYCE CODE

def adjust_lr_clevr(curr_los, prev_loss, curr_lr):
    loss_diff = prev_loss - curr_los
    not_improve = (
        (loss_diff < 0.015 and prev_loss < 0.5 and curr_lr > 0.00002) or
        (loss_diff < 0.008 and prev_loss < 0.15 and curr_lr > 0.00001) or
        (loss_diff < 0.003 and prev_loss < 0.10 and curr_lr > 0.000005))

    next_lr = curr_lr * cfg.TRAIN.SOLVER.LR_DECAY if not_improve else curr_lr
    return next_lr


def load_eval_data(max_num=0):
    load_eval_time = time.time()
    imdb_file = cfg.IMDB_FILE % cfg.TEST.SPLIT_REF
    data_reader = DataReader(
        imdb_file, shuffle=False, max_num=max_num,
        batch_size=cfg.TEST.BATCH_SIZE,
        vocab_question_file=cfg.VOCAB_QUESTION_FILE,
        T_encoder=cfg.T_ENCODER,
        vocab_answer_file=cfg.VOCAB_ANSWER_FILE,
        load_spatial_feature=True,
        spatial_feature_dir=cfg.SPATIAL_FEATURE_DIR,
        add_pos_enc=cfg.ADD_POS_ENC, img_H=cfg.IMG_H, img_W=cfg.IMG_W,
        pos_enc_dim=cfg.PE_DIM, pos_enc_scale=cfg.PE_SCALE)
    num_vocab = data_reader.batch_loader.vocab_dict.num_vocab
    num_choices = data_reader.batch_loader.answer_dict.num_vocab
    print('load_eval_time: ', time.time() - load_eval_time)
    return data_reader, num_vocab, num_choices


def run_eval_on_data(model, data_reader_eval, pred=False):
    model.eval()
    predictions = []
    correct, total, loss_sum, batch_num, AUC_sum, f1_sum, top_accuracy_sum = 0, 0, 0., 0, 0., 0., 0.
    for batch, _, _ in data_reader_eval.batches(one_pass=True):
        batch_res = model.run_batch(
            batch, train=False, run_vqa=False, run_ref=True)
        if pred:
            #BRYCE CODE
            predictions.extend({'image_ID': i, 'question_ID': q, 'accuracy': a, 'expression': e, 'expression_family': f, 'prediction': [x.tolist() for x in [b for b in p] if x[2]!=0], 'gt_boxes': [x.tolist() for x in [b for b in g] if x[2]!=0]}
                    for i, q, a, e, f, p, g in zip(batch['imageid_list'], batch['qid_list'], batch_res['top_accuracy_list'], batch['qstr_list'], batch['ref_list'], batch_res['bbox_predictions'], batch_res['gt_coords']))

            #print(predictions)
            #pause
            #BRYCE CODE
        correct += batch_res['bbox_num_correct']
        #BRYCE CODE
        total += batch_res['possible_correct_boxes']
        #BRYCE CODE
        loss_sum += batch_res['loss'].item()
        #AUC_sum += batch_res['pr_AUC']
        #f1_sum += batch_res['pr_f1']
        top_accuracy_sum += batch_res['top_accuracy']
        batch_num += 1
        #BRYCE CODE
        print('\rEval S %d: avgL=%.4f, avgA=%.4f, avgTopA=%.4f' % (total, loss_sum/batch_num, correct/total, top_accuracy_sum/batch_num), end='')
        #BRYCE CODE
    #print('')
    eval_res = {
        'correct': correct,
        'total': total,
        'box_accuracy': float(correct*1./total),
        'top_accuracy': top_accuracy_sum/batch_num,
        #'pr_AUC': float(AUC_sum/batch_num),
        #'pr_f1': float(f1_sum/batch_num),
        'loss': loss_sum/batch_num,
        'predictions': predictions}
    return eval_res


def dump_prediction_to_file(predictions, res_dir):
    pred_file = os.path.join(res_dir, 'pred_bbox_%s_%04d_%s.json' % (
        cfg.EXP_NAME, cfg.TEST.EPOCH, cfg.TEST.SPLIT_REF))
    with open(pred_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    print('predictions written to %s' % pred_file)


def train():
    data_reader_train, num_vocab, num_choices = load_train_data()
    data_reader_eval, _, _ = load_eval_data(max_num=cfg.TRAIN.EVAL_MAX_NUM)

    # Load model
    model = LCGNwrapper(num_vocab, num_choices)

    # Save snapshot
    snapshot_dir = os.path.dirname(cfg.SNAPSHOT_FILE % (cfg.EXP_NAME, 0))
    os.makedirs(snapshot_dir, exist_ok=True)
    with open(os.path.join(snapshot_dir, 'cfg.json'), 'w') as f:
        json.dump(cfg, f, indent=2)
    lr_start = cfg.TRAIN.SOLVER.LR
    if cfg.TRAIN.START_EPOCH > 0:
        print('resuming from epoch %d' % cfg.TRAIN.START_EPOCH)
        snapshot_file = cfg.SNAPSHOT_FILE % (
            cfg.EXP_NAME, cfg.TRAIN.START_EPOCH)
        model.load_state_dict(torch.load(snapshot_file))
        states_file = snapshot_file.replace('.ckpk', '') + '_states.npy'
        if os.path.exists(states_file):
            lr_start = np.load(states_file, allow_pickle=True)[()]['lr']
            print('recovered previous lr %.1e' % lr_start)
        else:
            print('could not recover previous lr')

    print('%s - train for %d epochs' % (cfg.EXP_NAME, cfg.TRAIN.MAX_EPOCH))
    run_train_on_data(
        model, data_reader_train, lr_start, run_eval=cfg.TRAIN.RUN_EVAL,
        data_reader_eval=data_reader_eval)
    print('%s - train (done)' % cfg.EXP_NAME)


def test():
    data_reader_eval, num_vocab, num_choices = load_eval_data()

    # Load model
    model = LCGNwrapper(num_vocab, num_choices)

    # Load test snapshot
    snapshot_file = cfg.SNAPSHOT_FILE % (cfg.EXP_NAME, cfg.TEST.EPOCH)
    model.load_state_dict(torch.load(snapshot_file))

    res_dir = cfg.TEST.RESULT_DIR % (cfg.EXP_NAME, cfg.TEST.EPOCH)
    vis_dir = os.path.join(
        res_dir, '%s_%s' % (cfg.TEST.VIS_DIR_PREFIX, cfg.TEST.SPLIT_REF))
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    pred = cfg.TEST.DUMP_PRED
    if not pred:
        print('NOT writing predictions (set TEST.DUMP_PRED True to write)')

    print('%s - test epoch %d' % (cfg.EXP_NAME, cfg.TEST.EPOCH))
    eval_res = run_eval_on_data(model, data_reader_eval, pred=pred)
    print('%s - test epoch %d: top accuracy = %.4f' % (
        cfg.EXP_NAME, cfg.TEST.EPOCH, eval_res['top_accuracy']))

    # write results
    if pred:
        dump_prediction_to_file(eval_res['predictions'], res_dir)
    eval_res.pop('predictions')
    res_file = os.path.join(res_dir, 'res_%s_%04d_%s.json' % (
        cfg.EXP_NAME, cfg.TEST.EPOCH, cfg.TEST.SPLIT_REF))
    with open(res_file, 'w') as f:
        json.dump(eval_res, f)


if __name__ == '__main__':
    if cfg.train:
        train()
    else:
        test()
