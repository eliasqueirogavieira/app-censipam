
import numpy as np
import rasterio
import skimage.io
import geopandas
import argparse


class Metrics:

    def __init__(self, num_classes: int, ignore_label: int, device) -> None:
        self.ignore_label = ignore_label
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def update(self, pred, target) -> None:

        #pred = pred.argmax(dim=1)
        keep1 = target != self.ignore_label
        #keep2 = pred != self.ignore_label
        #keep = keep1 & keep2
        
        keep = keep1

        self.hist += np.bincount(target[keep] * self.num_classes + pred[keep], minlength=self.num_classes**2).reshape(self.num_classes, self.num_classes)


    def compute_iou(self):
        ious = self.hist.diagonal() / (self.hist.sum(0) + self.hist.sum(1) - self.hist.diagonal())
        miou = ious[~np.isnan(ious)].mean()
        ious *= 100
        miou *= 100
        return ious.round(2).tolist(), round(miou, 2)

    def compute_iou_defor(self):
        ious = self.hist.diagonal() / (self.hist.sum(0) + self.hist.sum(1) - self.hist.diagonal())
        ious = ious[-1]
        ious *= 100
        return ious.round(2)

    def compute_f1(self):
        f1 = 2 * self.hist.diagonal() / (self.hist.sum(0) + self.hist.sum(1))
        mf1 = f1[~np.isnan(f1)].mean()
        f1 *= 100
        mf1 *= 100
        return f1.round(2).tolist(), round(mf1, 2)

    def compute_pixel_acc(self):
        acc = self.hist.diagonal() / self.hist.sum(1)
        macc = acc[~np.isnan(acc)].mean()
        acc *= 100
        macc *= 100
        return acc.round(2).tolist(), round(macc, 2)
    
    def precision(self):
        prec = self.hist.diagonal() / self.hist.sum(0)
        prec = 100 * prec[-1]
        return prec.round(2)

    def recall(self):
        rec = self.hist.diagonal() / self.hist.sum(1)
        rec = 100 * rec[-1]
        return rec.round(2)


def compute_metric(self, label, pred_mask):

        metrics = Metrics(2, -1, self.device)
        
        label = label.astype(np.int16)
        label[label == 128] = 1
        label[label == 255] = 2
        label = label - 1

        pred_mask = pred_mask.astype(np.int16)
        pred_mask[pred_mask == 128] = 0 # 1
        pred_mask[pred_mask == 255] = 1 # 2
        #pred_mask = pred_mask - 1

        metrics.update(pred_mask.flatten(), label.flatten())
        miou = metrics.compute_iou_defor()
        #acc, macc = metrics.compute_pixel_acc()
        #f1, mf1 = metrics.compute_f1()

        prec = metrics.precision()
        rec = metrics.recall()

        mf1 = 2*prec*rec / (prec + rec)
        mf1 = mf1.round(2)


        print(f"Global -- prec: {prec} \t rec: {rec} \t f1: {mf1} \t iou: {miou}")



def to_mask(tif_fname, geo_dtf):
    from rasterio import features

    with rasterio.open(tif_fname) as source:
        meta = source.meta
        shape = source.shape
        transform = source.transform
        band1 = source.read(1)
        band2 = source.read(2)
        mask1 = source.read_masks(1)
        mask2 = source.read_masks(2)
        tif_mask = mask1 & mask2

        
    mask = rasterio.features.geometry_mask(geo_dtf.geometry, shape, 
        transform, all_touched=False, invert=True).astype(np.uint8) + 1

    tif_m = tif_mask.astype(bool)

    mask[~tif_m] = 0
    
    mask[ mask == 1 ] = 128
    mask[ mask == 2 ] = 255

    skimage.io.imsave("mask_label.png", mask)

    meta['count'] = 1
    meta['dtype'] = 'uint8'
    meta['nodata'] = 0
    with rasterio.open('mask_label.tif', 'w', **meta) as dst:
        dst.write_band(1, mask)

    meta['count'] = 3
    meta['dtype'] = 'uint8'
    meta['nodata'] = 0
    with rasterio.open('masked.tif', 'w', **meta) as dst:
        dst.write_band(1, band1)
        dst.write_band(2, band2)
        dst.write_band(3, mask)

def evaluate_prediction(args):

    fname_raw_tif, fname_raw_shp, fname_pred_shp = 1
    
    shps_df = geopandas.read_file(fname_raw_shp)

    if shps_df.crs.to_epsg() != 4326:
        shps_df = shps_df.to_crs('EPSG:4326')
    
    deforest = shps_df.loc[ shps_df['CLASS_NAME'] == 'Deforestation' ]
    
    to_mask(fname_raw_tif, deforest)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-func',  type=str, default = 'evaluate_prediction')
    parser.add_argument('--raw_tif',  type=str, default = 'default')
    parser.add_argument('--raw_shp', type=str, default = 'default')
    parser.add_argument('--pred_tif', type=str, default = 'default')
    
    args = parser.parse_args()

    args.raw_tif = '/censipam_data/Datasets/sentinel_data/Sentinel_1A_10_615_21Mai2022_26Jun2022/S1A_IW_SLC__1SDV_20220521T094042_20220521T094110_043307_052BF5_7700_Orb_Cal_deb_ML_TF_Stack_EC.tif'
    args.raw_tif = '/censipam_data/Datasets/sentinel_data/Sentinel_1A_10_615_21Mai2022_26Jun2022/Sentinel_1A_10_615_21Mai2022_26Jun2022.shp'

    func = eval(args.func) 

    func(args)
    