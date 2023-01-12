
import argparse
from ast import arg
from glob import glob

#from torchmetrics import F1Score, Dice, JaccardIndex
from torchmetrics.functional import precision_recall, f1_score, dice, jaccard_index, confusion_matrix, stat_scores
from torchvision.io import read_image
import torch

from pathlib import Path
import csv

import matplotlib.pyplot as plt

import pandas
import numpy as np

# https://torchmetrics.readthedocs.io/en/stable/classification/precision_recall.html
# https://torchmetrics.readthedocs.io/en/stable/classification/jaccard_index.html
# https://torchmetrics.readthedocs.io/en/stable/classification/f1_score.html
# https://torchmetrics.readthedocs.io/en/stable/classification/dice.html


def main_driver(folder_label, folder_pred):

    lbl_fnames = glob(f'{folder_label}/*.png')
    pred_fnames = glob(f'{folder_pred}/*.png')

    lbl_fnames.sort()
    pred_fnames.sort()

    assert len(lbl_fnames) == len(pred_fnames), 'Different number of files'

    data_size = len(lbl_fnames)

    prec = recall = f1 = jaccard = dice_score = 0
    stat = None


    for l_fname, p_fname in zip(lbl_fnames, pred_fnames):

        lbl = read_image(l_fname).type(torch.long).view(-1)
        pred = read_image(p_fname).type(torch.long).view(-1)

        jaccard += jaccard_index(pred, lbl, num_classes = 256, average='micro', absent_score=0) / data_size
        dice_score += dice(pred, lbl) / data_size
        f1 += f1_score(pred, lbl, average = 'macro', num_classes = 256) / data_size

        tmp_a, tmp_b = precision_recall(pred, lbl, average = 'macro', num_classes = 256) 

        prec += tmp_a / data_size
        recall += tmp_b /data_size


        #c_matrix = confusion_matrix(pred, lbl, num_classes=256)
        if stat is None: stat = stat_scores(pred, lbl, reduce='macro', num_classes = 256)[-1]
        else: stat += stat_scores(pred, lbl, reduce='macro', num_classes = 256)[-1]
        

    #stat = stat.divide(stat[-1]).numpy()
    stat = stat.numpy()

    with open(Path(folder_pred) / 'results.csv', 'w') as pfile:
        wr = csv.writer(pfile)
        wr.writerow(['recall', 'precision', 'f1', 'jaccard', 'dice'])
        wr.writerow([recall.numpy(), prec.numpy(), f1.numpy(), jaccard.numpy(), dice_score.numpy()])
        
        # wr.writerow([recall.numpy(), prec.numpy(), f1.numpy(), jaccard.numpy(), dice_score.numpy(), 
        #             stat[0], stat[1], stat[2], stat[3] ])

    with open(Path(folder_pred) / 'stats.csv', 'w') as pfile:
        wr = csv.writer(pfile)
        wr.writerow(['tp', 'fp', 'tn', 'fn'])
        wr.writerow([stat[0], stat[1], stat[2], stat[3]])

    

        

def plot(folder_pred1, folder_pred2):

    # https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html

    #d1 = pandas.read_csv(Path(folder_pred1) / 'results.csv')
    #d2 = pandas.read_csv(Path(folder_pred2) / 'results.csv')

    d1 = pandas.read_csv(Path(folder_pred1) / 'stats.csv')
    d2 = pandas.read_csv(Path(folder_pred2) / 'stats.csv')

    d1.drop('tn', inplace=True, axis=1) # delete column
    d2.drop('tn', inplace=True, axis=1) # delete column

    d1.max().max()
    
    #d3 = pandas.read_csv(Path('/censipam_data/renam/datasets/sentinel_ready/eval/ddrnet') / 'results.csv')


    labels = d1.columns.values 
    x = np.arange(len(labels))
    width = 0.2  # the width of the bars

    prec = 0.001

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, np.floor(d1.iloc[0].values / prec) * prec, width, label=Path(folder_pred1).stem)     
    rects2 = ax.bar(x + width, np.floor(d2.iloc[0].values / prec) * prec, width, label=Path(folder_pred2).stem)
    #rects3 = ax.bar(x + width*2, np.floor(d1.iloc[0].values / prec) * prec, width, label='DDRNet')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Value')
    ax.set_title('Performance metrics')
    ax.set_xticks(x, labels)
    ax.legend(loc='upper center')
    
    ax.set_ylim([0, 3.5e7])

    #ax.bar_label(rects1, padding=5)
    #ax.bar_label(rects2, padding=5)
    #ax.bar_label(rects3, padding=5)

    #fig.tight_layout()

    plt.show()

    #plt.bar(d1.columns.values ,d1.iloc[0].values)
    #plt.bar(d2.columns.values ,d2.iloc[0].values)

    plt.savefig('plot.png')

    # with open(Path('/censipam_data/renam/datasets/sentinel_ready/eval/elias') / 'results.csv', 'r') as pfile:
    #     data1 = csv.reader(pfile)

    #     for i, row in enumerate(data1):   
    #         print(', '.join(row))

    # with open(Path('/censipam_data/renam/datasets/sentinel_ready/eval/ddrnet') / 'results.csv', 'r') as pfile:
    #     data1 = csv.reader(pfile)

    #     for i, row in enumerate(data1):
    #         print(', '.join(row))







def parse_params():

    parser = argparse.ArgumentParser()
    parser.add_argument('-folder_labels', type=str, default='/censipam_data/renam/datasets/sentinel_ready/eval/labels')
    #parser.add_argument('-folder_preds', type=str, default='/censipam_data/renam/datasets/sentinel_ready/eval/ddrnet')
    #parser.add_argument('-folder_preds', type=str, default='/censipam_data/renam/datasets/sentinel_ready/eval/ddrnet')

    parser.add_argument('-folder_preds', action='store', type=str, nargs='*')

    args = parser.parse_args()

    return args


def main(args):

    print("Computing performance metrics ...")
    for p in args.folder_preds:
        main_driver(args.folder_labels, p)
        print("--done")
    
    plot(args.folder_preds[0], args.folder_preds[1])
    print("Done")



if __name__ == '__main__':


    args = parse_params()
    main(args)

    

