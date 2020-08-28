import json
import argparse
import sys
import os.path as osp
from PIL import Image
import numpy as np
from numpy import asarray
import matplotlib.pyplot as plt
here = osp.dirname(osp.abspath(__file__))
import imgviz


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default="/u/byahn2/LCGN_Sets/exp_clevr/results/lcgn_ref/0030/pred_bbox_lcgn_ref_0030_locplus_val.json",
    help="The location of the file containing the results of the evaluation")
parser.add_argument('--image_height', default=224, type=int)
parser.add_argument('--image_width', default=224, type=int)

# information in data includes: image_ID, question_ID, accuracy, expression, expression_family, prediction (of bounding boxes), and gt_boxes (ground truth bounding boxes)

def main(args):

    with open(args.data_path, 'r') as f:
        data = json.load(f)
    print('data: ', len(data))
    image_dir = '/u/byahn2/LCGN_Sets/exp_clevr/clevr_locplus_dataset/images/val/'
    img_size = (args.image_height, args.image_width)
    for i in range(len(data)):
        num_zeros = 10 - len(data[i]["image_ID"])
        z = ''
        for n in range(num_zeros):
            z = z + '0'
        image_name = data[i]["image_ID"][0:4] + z + data[i]["image_ID"][4:]
        file_name = 'CLEVR_' + image_name + '.png'
        input_path = image_dir + file_name
        img = Image.open(input_path)
        img = img.convert('RGB')
        #img = img.resize(img_size, resample=Image.BICUBIC)
        img = asarray(img)

        boxes = []
        labels = []
        captions = []
        count = 0
        for j in range(len(data[i]["prediction"])):
            original_box = np.array(data[i]["prediction"][j])
            new_box = np.zeros(4)
            new_box[0] = original_box[1] # y1 = y
            new_box[1] = original_box[0] # x1 = x
            new_box[2] = original_box[1] + original_box[2] # y2 = y + w
            new_box[3] = original_box[0] + original_box[3] # x2 = x + h
            #y1, x1, y2, x2   left top - right bottom
            # x, y, w, h where x,y are for left top
            boxes.append(new_box)
            labels.append(int(count+j))
            captions.append('pred')
        count = 0
        for j in range(len(data[i]["gt_boxes"])):
            original_box = np.array(data[i]["gt_boxes"][j])
            new_box = np.zeros(4)
            new_box[0] = original_box[1] # y1 = y
            new_box[1] = original_box[0] # x1 = x
            new_box[2] = original_box[1] + original_box[2] # y2 = y + w
            new_box[3] = original_box[0] + original_box[3] # x2 = x + h
            boxes.append(new_box)
            labels.append(int(count+j))
            captions.append('gt')
        
        print('image: ', img.shape)
        print('labels: ', len(labels)) # int 
        print('captions: ', len(captions)) # string
        print('bboxes: ', len(boxes), ' ', boxes)

        bboxviz = imgviz.instances2rgb(image=img, bboxes=boxes, labels=labels, captions=captions)

        out_file = osp.join('/u/byahn2/LCGN_Sets/exp_clevr/results/visuals/val' + '_' + str(i) + '.jpg')
        imgviz.io.imsave(out_file, bboxviz)

    #Evaluate accuracy based on question type
    num_families = 18
    n_bins = 20
    exp_acc = [np.empty(1)]*num_families
    ref_idx = {1: 'same_relate', 2: 'same_relate_b', 3: 'zero_hop', 4: 'zero_hop_b', 5:'one_hop', 6: 'one_hop_b', 7: 'two_hop', 8: 'two_hop_b', 9: 'three_hop', 10: 'three_hop_b', 11: 'single_and', 12: 'single_and_b', 13: 'single_or', 14: 'single_or_b', 15: 'single_not', 16: 'single_not_b', 17: 'count_returns',18: 'check_farthest'}
    print('len_data: ', len(data))
    for f in range(num_families):
        exp_acc[f] = np.empty(1, dtype=np.float32)
    for i in range(len(data)):
        f = int(data[i]['expression_family'])-1
        exp_acc[f] = np.append(exp_acc[f], data[i]['accuracy'])
    acc_means = np.zeros(num_families)
    for f in range(num_families):
        acc_means[f] = np.mean(exp_acc[f])
        plt.hist(exp_acc, n_bins)
        plt.xlabel('Accuracy')
        print(f+1)
        title_string = 'Expression_family: ' + ref_idx[f+1]
        plt.title(title_string)
        out_file = '/u/byahn2/LCGN/exp_clevr/results/exp_eval/exp_' + ref_idx[f+1] + '.jpg'
        plt.savefig(out_file)
    print('mean accuracy: ', acc_means*100)



if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
