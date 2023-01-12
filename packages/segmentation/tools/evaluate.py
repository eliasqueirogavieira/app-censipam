
import argparse
from glob import glob

#from torchmetrics import F1Score, Dice, JaccardIndex
from torchmetrics.functional import precision_recall, f1_score, dice, jaccard_index
from torchvision.io import read_image
import torch

# https://torchmetrics.readthedocs.io/en/stable/classification/precision_recall.html
# https://torchmetrics.readthedocs.io/en/stable/classification/jaccard_index.html
# https://torchmetrics.readthedocs.io/en/stable/classification/f1_score.html
# https://torchmetrics.readthedocs.io/en/stable/classification/dice.html

    #jaccard = JaccardIndex(num_classes=256)
    #dice = Dice(average='micro')
    #f1 = F1Score(num_classes=256) # mdmc_average='samplewise'

def main_driver(args):

    lbl_fnames = glob(f'{args.folder_labels}/*.png')
    pred_fnames = glob(f'{args.folder_preds}/*.png')

    lbl_fnames.sort()
    pred_fnames.sort()

    assert len(lbl_fnames) == len(pred_fnames), 'Different number of files'

    data_size = len(lbl_fnames)

    p = r = f1 = j = d = 0

    for l_fname, p_fname in zip(lbl_fnames, pred_fnames):

        lbl = read_image(l_fname).type(torch.long).view(-1)
        pred = read_image(p_fname).type(torch.long).view(-1)

        j += jaccard_index(pred, lbl, num_classes = 256, average='micro', absent_score=0) / data_size
        d += dice(pred, lbl) / data_size
        f1 += f1_score(pred, lbl, average = 'macro', num_classes = 256) / data_size

        a, b = precision_recall(pred, lbl, average = 'macro', num_classes = 256) 

        p += a / data_size
        r += b /data_size

        # f1_val = f1(pred, lbl)
        # dice_val = dice(pred, lbl)
        # prec_rec = precision_recall(pred, lbl, average='macro', num_classes = 256)

        #metrics.append([p, r, f1, j, d])

        
    print('recall: {0} \t prec: {1} \t f1: {2} \t jaccard: {3} \t dice: {4}'.format(r, p, f1, j, d))
    #print(metrics)
        
        
        #jaccard(pred, lbl)

        







def parse_params():

    parser = argparse.ArgumentParser()
    parser.add_argument('-folder_labels', type=str, default='/censipam_data/renam/datasets/sentinel_ready/eval/labels')
    parser.add_argument('-folder_preds', type=str, default='/censipam_data/renam/datasets/sentinel_ready/eval/ddrnet')
    args = parser.parse_args()

    return args



if __name__ == '__main__':


    args = parse_params()

    main_driver(args)


