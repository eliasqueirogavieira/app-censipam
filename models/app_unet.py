from curses import meta
from turtle import back
from cv2 import dft, merge
import torch
import re
import numpy as np
from os.path import exists
from torch import Tensor, le, merge_type_from_type_comment
from torch.nn import functional as F
from semseg.utils.visualize import draw_text
from semseg.utils.utils import timer
from tqdm import tqdm
from .sfnet import sfnet
from .imodel import NNModel
from .imodel import pred2Shapefile
from skimage.io import imsave
import matplotlib.pyplot as plt
import rasterio
import geopandas
from geopandas.tools import overlay
from tqdm import tqdm
from shutil import rmtree
import os
from .unet import model
from pathlib import Path
from qgis.core import *
from qgis.PyQt.QtCore import QVariant
from osgeo import ogr, osr

def initialize_qgis():
    QgsApplication.setPrefixPath('/home/eliasqueiroga/anaconda3/envs/appcensipam', True)
    qgs = QgsApplication([], False)
    qgs.initQgis()
    print("Success! QGIS Initialized")

def add_area_attr(shapefile_path, attr_name):
    dataSource = ogr.GetDriverByName("ESRI Shapefile").Open(shapefile_path, 1)
    layer = dataSource.GetLayer()
    if layer.GetLayerDefn().GetFieldIndex(attr_name)==-1:
        new_field = ogr.FieldDefn(attr_name, ogr.OFTReal)
        layer.CreateField(new_field)

    using_latlong = layer.GetSpatialRef().EPSGTreatsAsLatLong()
    if using_latlong:
        tgt_srs = osr.SpatialReference()
        tgt_srs.ImportFromEPSG(3857)
        transform = osr.CoordinateTransformation(layer.GetSpatialRef(), tgt_srs)

    for feature in layer:
        geom = feature.GetGeometryRef()
        if hasattr(geom,'GetArea'):
            if using_latlong:
                geom2 = geom.Clone()
                geom2.Transform(transform)
                area = geom2.GetArea()/1e6
            else:
                area = geom.GetArea()/1e6
            feature.SetField(attr_name, area)
            layer.SetFeature(feature)
    return area, feature

class ModelUNet(NNModel):
    def __init__(self, config) -> None:
        super().__init__()
        
        self.config = config
        self.backbone = config.MODEL.backbone
        self.obj_list = config.MODEL.obj_list
        self.num_classes = 1 # len(obj_list)
        self.checkpoint = config.MODEL.checkpoint

        self.device = config.MODEL.device
        self.batch_sz = config.MODEL.batch_size
        self.threshold = config.MODEL.threshold

        self.width = config.MODEL.patch_size
        self.height = config.MODEL.patch_size
        self.offset = self.height // 2
        
        self.palette = [0], [255]
        self.palette = torch.tensor(self.palette)
        self.labels = ['background', 'deforestation' ]

        self.model = model.UNet()      
 

    def load(self):
        assert(exists(self.checkpoint)),f'App unet: checkpoint does not exist'
        self.model.load_state_dict(torch.load(self.checkpoint, map_location='cpu'))
        #self.model = torch.load(self.checkpoint)
        self.model = self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def postprocess(self, seg_map: Tensor) -> Tensor:
        seg_map = torch.sigmoid(seg_map)
        seg_map = (seg_map>self.threshold).cpu().to(int)
        seg_image = self.palette[seg_map].squeeze()
        return seg_image
    
    @torch.inference_mode()
    def postprocess_score(self, seg_map: Tensor) -> Tensor:
        seg_map = torch.sigmoid(seg_map)
        seg_map = (seg_map).cpu().to(float)        
        seg_image = seg_map.squeeze()
        return seg_image
 
    def __preprocess(self, image: Tensor) -> Tensor:
        image = image / 255 # divide by 255, norm and add batch dim
        image = image.to(self.device)
        return image
    
    @torch.inference_mode() #@timer
    def model_forward(self, img: Tensor) -> Tensor:
        return self.model(img)

    def __predict(self, image: Tensor, overlay: bool = False) -> Tensor:
        
        img = self.__preprocess(image)
        seg_map = self.model_forward(img)
        
        #seg_map = self.postprocess(seg_map)
        seg_map = self.postprocess_score(seg_map)

        return seg_map

    def save2file(self, batch_idx, b_before, b_after, transforms, base_fnames, meta):

        assert b_before.shape == b_after.shape

        b_before = b_before.astype(np.uint8)
        b_after = b_after.astype(np.uint8)

        diff = np.zeros(b_before.shape)        
        diff[ b_before != b_after ] = 255

        for i in range(b_before.shape[0]):

            meta['count'] = 1
            meta['transform'] = transforms[i]
            fname = f"before/{base_fnames[i]}.tif"

            with rasterio.open(fname, 'w', **meta) as dst:
                dst.write_band(1, b_before[i])

            imsave(f"before/{base_fnames[i]}.png", b_before[i])
            imsave(f"after/{base_fnames[i]}.png", b_after[i])


    def save2file_deb(self, b_before, b_after, transforms, base_fnames, meta, global_shp = None):

        assert b_before.shape == b_after.shape
        if not len(b_before.shape) == 3:
            b_before = np.expand_dims(b_before, 0)
            b_after = np.expand_dims(b_after, 0)

        b_before = b_before.astype(np.uint8)
        b_after = b_after.astype(np.uint8)
      
        diff_deforest = (b_after > b_before).astype(np.uint8)

        for i in range(b_before.shape[0]):
            
            meta['count'] = 1
            meta['transform'] = transforms[i]
            fname = "tmp"
            
            with rasterio.open(f"{fname}.tif", 'w', **meta) as dst:
                dst.write_band(1, diff_deforest[i])

            # gdal_polygonize.py -8 crop_2_8.tif shape.shp
            pred2Shapefile(f"{fname}.tif", f"{fname}.shp")
            shps_df = geopandas.read_file(f"{fname}.shp")

            if not isinstance(global_shp, geopandas.GeoDataFrame):
                global_shp = shps_df
            else:
                global_shp = global_shp.append(shps_df)

        return global_shp, diff_deforest

	
    def predict_single(self, patches):

        batch_size = self.batch_sz
        nb_batch = np.ceil(len(patches.patches) / batch_size).astype(int)
        offset_batch = 0
        df_preds = None
        
        for batch_idx in range(nb_batch):

            offset_batch = batch_idx * self.batch_sz

            if (len(patches.patches) < self.batch_sz):
                block_width = len(patches.patches)
            else:
                block_width = min(self.batch_sz, len(patches.patches) - self.batch_sz)

            batch = patches.patches[offset_batch: offset_batch + block_width]
            transforms = patches.transforms[offset_batch: offset_batch + block_width]
            base_fnames = patches.base_fnames[offset_batch: offset_batch + block_width]

            batch = np.stack(batch, axis = 0)
            
            batch_before = batch[:, 0:2, ...]
            batch_after = batch[:, 2:, ...]
            del batch

            batch_after = torch.from_numpy(batch_after) # torch.tensor(batch)
            batch_before = torch.from_numpy(batch_before)

            pred_after = self.__predict(batch_after)
            pred_before = self.__predict(batch_before)

            pred_after =  pred_after.detach().numpy()
            pred_before =  pred_before.detach().numpy()

            df_preds, _ = self.save2file_deb(pred_before, pred_after, transforms, base_fnames, patches.meta, df_preds)

        return df_preds


    def __predict_raster_grid(self, patches, stacked_fname):

        with rasterio.open(stacked_fname) as src:
            w = src.width
            h = src.height        
        
        #preds_complete  = [ np.zeros((h, w), dtype = np.uint8), np.zeros((h, w), dtype = np.uint8) ]
        preds_complete  = [ np.zeros((h, w), dtype = np.float16) - 1, np.zeros((h, w), dtype = np.float16) - 1]
        nb_batch = np.ceil(len(patches.patches) / self.batch_sz).astype(int)

        for batch_idx in tqdm(range(nb_batch)):

            offset_batch = batch_idx * self.batch_sz         
            block_width = min(self.batch_sz, len(patches.patches) - offset_batch)
            if block_width == 1:
                return preds_complete
            batch = patches.patches[offset_batch: offset_batch + block_width]
            crop_windows = patches.crop_windows[offset_batch: offset_batch + block_width]
            batch = np.stack(batch, axis = 0)
            
            batch_before = batch[:, 0:2, ...]
            batch_after = batch[:, 2:, ...]
            del batch

            batch_after = torch.from_numpy(batch_after) # torch.tensor(batch)
            batch_before = torch.from_numpy(batch_before)

            pred_after = self.__predict(batch_after).detach().numpy().astype(np.float16)
            pred_before = self.__predict(batch_before).detach().numpy().astype(np.float16)

            pred_before  = np.split(pred_before, block_width)
            pred_after  = np.split(pred_after, block_width)

            pred_before = [np.squeeze(p) for p in pred_before]
            pred_after = [np.squeeze(p) for p in pred_after]

            preds_complete = self.__gather_blocks_together([pred_before, pred_after], crop_windows, preds_complete)
            
        return preds_complete

    
    def __gather_blocks_together(self, batch, c_windows, pred_mask):

        for i in range(len(batch[0])):

            w = c_windows[i]
            pred_mask[0][w.row_off:(w.row_off + w.height), w.col_off :(w.col_off + w.width)] = \
                batch[0][i][0:w.height, 0:w.width]
            
            pred_mask[1][w.row_off:(w.row_off + w.height), w.col_off :(w.col_off + w.width)] = \
                batch[1][i][0:w.height, 0:w.width]
        
        return pred_mask
            
    def predict(self, patch_grids, stacked_fname, out_folder, orig_folder):

        print(f'\t\tRunning the model...')
        
        #self.predict_vector(patch_grids, stacked_fname, out_folder)
        self.predict_raster(patch_grids, stacked_fname, out_folder, orig_folder)
        
        print(f'\t\tDone')

    
    def predict_vector(self, patch_grids, stacked_fname, out_folder):

        dtf_intersec = None
        for patch_list in patch_grids:

            dtf_pred = self.predict_single(patch_list)
            
            if dtf_intersec is None:
                dtf_intersec = dtf_pred
            else:
                dtf_pred = dtf_pred.rename({'HA': 'tmp'}, axis=1)
                dtf_intersec = overlay(dtf_intersec, dtf_pred, how='intersection', keep_geom_type = True) #  make_valid = True
                dtf_intersec = dtf_intersec.drop(columns = 'tmp')
        
        out_folder = out_folder / 'preds.shp'
        dtf_intersec.to_file(out_folder)       

    
    def predict_raster(self, patch_grids, stacked_fname, out_folder, orig_folder):

        with rasterio.open(stacked_fname) as src:
            self.meta = src.meta
            self.meta['count'] = 1

        preds_before = []
        preds_after = []
        
        for _, grid in enumerate(patch_grids):
            
            preds_tmp = self.__predict_raster_grid(grid, stacked_fname)
            preds_before.append( preds_tmp[0] )
            preds_after.append( preds_tmp[1] )
        
        preds_before = np.stack(preds_before, axis = 0)
        preds_after = np.stack(preds_after, axis = 0)
        preds_before, preds_after, diff_pred = self.__combine_preds_score_sum(preds_before, preds_after) # should be binary
        date = re.findall('[0-9]{2}[a-z]+[0-9]{4}', orig_folder, flags=re.IGNORECASE)
        try:
            for i in range(len(date)):
                date[i] = date[i][0:2] + ' ' + date[i][2:5] + ' ' + date[i][5:]
        except:
            pass
        self.__save_PredToShpfile(out_folder / "pred.tif", diff_pred)
        df = geopandas.read_file(out_folder / "pred.shp")
        date_list = []
        for i in range(len(df['geometry'])): 
            date_list.append(str(date))
        df['date'] = date_list
        df_copy = df.copy()
        df_copy = df_copy.to_crs({'proj':'cea'})
        df['poly_area'] = df_copy['geometry'].area/ 10**6
        area_total = 0
        for index in range(len(df['poly_area'])): 
            area_total += df['poly_area'][index]
        area_total_list = []
        for i in range(len(df['geometry'])): 
            area_total_list.append(area_total)
        df['total_area'] = area_total_list
        df.to_file(str(out_folder / 'pred.shp'))
        #initialize_qgis()
        #area, feature = add_area_attr(str(out_folder / 'pred.shp'), 'Area_km2')
        #qgs.exitQgis()
        if self.config.OUTPUT.save_partial_results:
            self.__save_PredToShpfile(out_folder / 'pred_before.tif', self.__toBinary(preds_before))
            self.__save_PredToShpfile(out_folder / 'pred_after.tif', self.__toBinary(preds_after))
        
        with open( Path(out_folder).parent.parent / 'exec_report.txt', 'a') as report:
            report.write(f'Inference on : {Path(stacked_fname).parts[-2]}  > Ok \n')

    def __save_PredToShpfile(self, fname_out, pred):

        try:
            with rasterio.open(fname_out, 'w', **self.meta) as dst:
                dst.write_band(1, pred)
            
            pred2Shapefile(str(fname_out), str(fname_out.with_suffix('.shp')))
            if not self.config.OUTPUT.keep_pred_tif:
                os.remove(fname_out)

        except:
            print("Could not write the prediction shapefile to disk")
    
    def __toBinary(self, pred):
        pred[ pred <= 128 ] = 0 # must come before
        pred[ pred > 128 ] = 1 # must come before
        return pred

    
    def combine_preds_sum(self, preds_before, preds_after):

        threshold = 3*255

        preds_before = preds_before.astype(np.uint16)
        preds_after = preds_after.astype(np.uint16)
        
        preds_after = preds_after.sum(axis=0).astype(np.uint16)
        preds_before = preds_before.sum(axis=0).astype(np.uint16)

        preds_before[(preds_before > 0) & (preds_before < threshold) ] = 0
        preds_before[preds_before >= threshold ] = 255

        preds_before = preds_before.astype(np.uint8)

        preds_after[(preds_after > 0) & (preds_after < threshold) ] = 0
        preds_after[preds_after >= threshold ] = 255
        preds_after = preds_after.astype(np.uint8)

        preds_before = self.morph(preds_before)
        preds_after = self.morph(preds_after)
        
        #mask_zero = (preds_before == 0) & (preds_after == 0)
        
        diff_deforest = (preds_after > preds_before).astype(np.uint8)
        return diff_deforest
    
    def __combine_preds_score_sum(self, preds_before, preds_after):

        threshold = self.config.OUTPUT.threshold
        
        mask_before = preds_before < 0 
        mask_after = preds_after < 0
        mask = (mask_before.sum(axis=0) == preds_before.shape[0]) | (mask_after.sum(axis=0) == mask_after.shape[0])

        preds_before[mask_before] = 0
        preds_after[mask_after] = 0
        
        d = (~mask_before).sum(axis=0).astype(np.float16)
        d [d < 1] = 1e-6
        preds_before = (preds_before).sum(axis=0) / d

        d = (~mask_after).sum(axis=0).astype(np.float16)
        d [d < 1] = 1e-6
        preds_after = (preds_after).sum(axis=0) / d
               
        preds_before[ preds_before >= threshold ] = 255 # must come before
        preds_before[ preds_before < threshold ] = 128
        preds_before[mask] = 0
        preds_before = preds_before.astype(np.uint8)

        preds_after[ preds_after >= threshold ] = 255 # must come before
        preds_after[ preds_after < threshold ] = 128
        preds_after[mask] = 0
        preds_after = preds_after.astype(np.uint8)
        
        if self.config.OUTPUT.apply_opening: 
            preds_before = self.morph(preds_before)
            preds_after = self.morph(preds_after)
                
        diff_deforest = (preds_after > preds_before).astype(np.uint8)

        return preds_before, preds_after, diff_deforest


    def morph(self, pred):
        """ Perform opening (erosion then dilation)
        
        pred: should take values from [0, 128, 255] 
                0: nodata
                128: no_deforestation
                255: deforestation
        
        return: pred resulted after the opening operation
        """ 
        from skimage.morphology import opening, square, closing, erosion, dilation

        mask = pred == 0
        pred [ pred == 128] = 0
        pred [ pred == 255] = 1

        pred = erosion(pred, square(5))
        pred = dilation(pred, square(7))

        pred[ pred == 0 ] = 128
        pred[ pred == 1 ] = 255
        pred[ mask ] = 0

        return pred