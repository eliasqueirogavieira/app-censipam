
import numpy as np
import rasterio
import skimage.io
import argparse


class Metrics:

    def __init__(self, num_classes: int, ignore_label: int) -> None:
        self.ignore_label = ignore_label
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def update(self, pred, target) -> None:

        keep = target != self.ignore_label
        self.hist += np.bincount(target[keep] * self.num_classes + pred[keep], minlength=self.num_classes**2).reshape(self.num_classes, self.num_classes)


    # def compute_iou(self):
    #     ious = self.hist.diagonal() / (self.hist.sum(0) + self.hist.sum(1) - self.hist.diagonal())
    #     miou = ious[~np.isnan(ious)].mean()
    #     ious *= 100
    #     miou *= 100
    #     return ious.round(2).tolist(), round(miou, 2)

    def compute_iou_defor(self):
        ious = self.hist.diagonal() / (self.hist.sum(0) + self.hist.sum(1) - self.hist.diagonal())
        ious = ious[-1]
        ious *= 100
        return ious.round(2)

    # def compute_f1(self):
    #     f1 = 2 * self.hist.diagonal() / (self.hist.sum(0) + self.hist.sum(1))
    #     mf1 = f1[~np.isnan(f1)].mean()
    #     f1 *= 100
    #     mf1 *= 100
    #     return f1.round(2).tolist(), round(mf1, 2)

    # def compute_pixel_acc(self):
    #     acc = self.hist.diagonal() / self.hist.sum(1)
    #     macc = acc[~np.isnan(acc)].mean()
    #     acc *= 100
    #     macc *= 100
    #     return acc.round(2).tolist(), round(macc, 2)
    
    def precision(self):
        prec = self.hist.diagonal() / self.hist.sum(0)
        prec = 100 * prec[-1]
        return prec.round(2)

    def recall(self):
        rec = self.hist.diagonal() / self.hist.sum(1)
        rec = 100 * rec[-1]
        return rec.round(2)


def compute_metric(label, pred_mask):

        metrics = Metrics(2, -1)
        
        label = label.astype(np.int16)
        label[label == 128] = 1
        label[label == 255] = 2
        label = label - 1

        pred_mask = pred_mask.astype(np.int16)
        pred_mask[pred_mask == 128] = 0 # 1
        pred_mask[pred_mask == 255] = 1 # 2
        
        metrics.update(pred_mask.flatten(), label.flatten())
        miou = metrics.compute_iou_defor()
        
        prec = metrics.precision()
        rec = metrics.recall()

        mf1 = 2*prec*rec / (prec + rec)
        mf1 = mf1.round(2)

        print(f"Global -- prec: {prec} \t rec: {rec} \t f1: {mf1} \t iou: {miou}")


def evaluate_prediction(args):

    with rasterio.open(args.label) as src:
        meta_label = src.meta
        label = src.read(1)
    
    with rasterio.open(args.pred) as src:
        meta_pred = src.meta
        pred = src.read(1)
    
    compute_metric(label, pred)


if __name__ == '__main__':
    '''Compute the semantic segmentation performance metrics. 
    Input tif files must comply to the following convention
    Only 3 values as allowed
    0: nodata
    128: no_deforestation
    255: deforestation
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--label',  type=str)
    parser.add_argument('--pred', type=str)
    args = parser.parse_args()

    evaluate_prediction(args) 
    
    