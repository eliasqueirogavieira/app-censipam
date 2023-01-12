from curses import meta
from turtle import back
from cv2 import dft, merge
import torch
import numpy as np
from os.path import exists
from torch import Tensor, le, merge_type_from_type_comment

from torch.nn import functional as F
from semseg.utils.visualize import draw_text

#from os.path import exists

#from skimage.io import imsave
#from torchvision.ops.boxes import batched_nms
#from shapely.geometry import Polygon

from semseg.utils.utils import timer
from tqdm import tqdm

# from .edet.backbone import EfficientDetBackbone 
# from .edet.utils import BB_STANDARD_COLORS, standard_to_bgr, get_index_label

from .sfnet import sfnet

from .imodel import NNModel
from .imodel import pred2Shapefile

from skimage.io import imsave

import matplotlib.pyplot as plt

import rasterio
#import skimage.io

import geopandas
from geopandas.tools import overlay
from tqdm import tqdm
from shutil import rmtree
import os



class ModelSFNet(NNModel):


    def __init__(self, config) -> None:
        super().__init__()
        
        self.config = config
        self.backbone = config.MODEL.backbone
        self.obj_list = config.MODEL.obj_list
        self.num_classes = len(self.obj_list) # len(obj_list)
        self.checkpoint = config.MODEL.checkpoint

        self.device = config.MODEL.device
        self.batch_sz = config.MODEL.batch_size
        self.threshold = config.MODEL.threshold

        self.width = config.MODEL.patch_size
        self.height = config.MODEL.patch_size
        self.offset = self.height // 2
        
        self.palette = [1], [255]
        self.palette = torch.tensor(self.palette)
        self.labels = ['background', 'deforestation' ]

        self.model = sfnet.SFNet(self.backbone, self.num_classes)        

    def load(self):
        assert(exists(self.checkpoint)),f'App sfnet: checkpoint does not exist'
        self.model.load_state_dict(torch.load(self.checkpoint, map_location='cpu'))
        self.model = self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def postprocess(self, seg_map: Tensor) -> Tensor:
                
        seg_map = seg_map.softmax(dim=1)

        inds = seg_map[:, 1:, :, :] < self.threshold
        tmp = torch.broadcast_to(torch.zeros(seg_map.shape[-2:]) < 0, (seg_map.shape[0], 1, -1, -1))
        inds = torch.cat([tmp, inds], 1)
        seg_map[inds] = 0.0
        
        seg_map = seg_map.argmax(dim=1).cpu().to(int)
        seg_image = self.palette[seg_map].squeeze()
        return seg_image
    
    @torch.inference_mode()
    def postprocess_debug(self, seg_map: Tensor) -> Tensor:
                
        seg_map = seg_map.softmax(dim=1)

        #inds = seg_map[:, 1:, :, :] < self.threshold
        #tmp = torch.broadcast_to(torch.zeros(seg_map.shape[-2:]) < 0, (seg_map.shape[0], 1, -1, -1))
        #inds = torch.cat([tmp, inds], 1)
        #seg_map[inds] = 0.0
        
        seg_image = seg_map[:, 1, ...].cpu()
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
        
        seg_map = self.postprocess(seg_map)
        #seg_map = self.postprocess_debug(seg_map)

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


    def predict_raster_grid(self, patches, stacked_fname):

        with rasterio.open(stacked_fname) as src:
            w = src.width
            h = src.height        
        
        preds  = [ np.zeros((h, w), dtype = np.uint8), np.zeros((h, w), dtype = np.uint8) ]
        #preds  = [ np.zeros((h, w), dtype = np.float16) - 1, np.zeros((h, w), dtype = np.float16) - 1 ]

        nb_batch = np.ceil(len(patches.patches) / self.batch_sz).astype(int)
        
        for batch_idx in tqdm(range(nb_batch)):

            offset_batch = batch_idx * self.batch_sz         
            block_width = min(self.batch_sz, len(patches.patches) - offset_batch)

            batch = patches.patches[offset_batch: offset_batch + block_width]
            crop_windows = patches.crop_windows[offset_batch: offset_batch + block_width]
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
        
            # debug
            pred_before = pred_before.astype(np.float16)
            pred_after = pred_after.astype(np.float16)

            pred_before  = np.split(pred_before, block_width)
            pred_after  = np.split(pred_after, block_width)

            ## preds = self.__gather_blocks_together([pred_before, pred_after], crop_windows, preds)
            preds = self.__gather_blocks_together([pred_before, pred_after], crop_windows, preds)
            
        return preds

    
    def __gather_blocks_together(self, batch, c_windows, pred_mask):

        for i in range(len(batch[0])):
            w = c_windows[i]
            pred_mask[0][w.row_off:(w.row_off + w.height), w.col_off :(w.col_off + w.width)] = batch[0][i]
            pred_mask[1][w.row_off:(w.row_off + w.height), w.col_off :(w.col_off + w.width)] = batch[1][i]
        
        return pred_mask
            
    def predict(self, patch_grids, stacked_fname, out_folder):

        print(f'\t\tRunning the model...')
        
        #self.predict_vector(patch_grids, stacked_fname, out_folder)
        self.predict_raster(patch_grids, stacked_fname, out_folder)
        
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

    
    def predict_raster(self, patch_grids, stacked_fname, out_folder):

        preds_before = []
        preds_after = []
        
        for _, patch_list in enumerate(patch_grids):
            
            preds_tmp = self.predict_raster_grid(patch_list, stacked_fname)
            preds_before.append( preds_tmp[0] )
            preds_after.append( preds_tmp[1] )
        
        preds_before = np.stack(preds_before, axis = 0)
        preds_after = np.stack(preds_after, axis = 0)
        
        diff_pred = self.combine_preds(preds_before, preds_after) # should be binary

        with rasterio.open(stacked_fname) as src:
            meta = src.meta

        meta['count'] = 1
        fname = out_folder / "pred"
        with rasterio.open(f"{fname}.tif", 'w', **meta) as dst:
            dst.write_band(1, diff_pred)
        
        pred2Shapefile(f"{fname}.tif", f"{fname}.shp")
        os.remove(f"{fname}.tif")

        # meta['count'] = 4
        # meta['dtype'] = rasterio.float32
        # meta['nodata'] = -1.0

        # with rasterio.open("before.tif", 'w', **meta) as dst:
        #     dst.write_band(1, preds_before[0])
        #     dst.write_band(2, preds_before[1])
        #     dst.write_band(3, preds_before[2])
        #     dst.write_band(4, preds_before[3])
        
        # with rasterio.open("after.tif", 'w', **meta) as dst:
        #     dst.write_band(1, preds_after[0])
        #     dst.write_band(2, preds_after[1])
        #     dst.write_band(3, preds_after[2])
        #     dst.write_band(4, preds_after[3])

    
    def combine_preds(self, preds_before, preds_after):

        threshold = 3*255

        preds_before = preds_before.astype(np.uint16)
        preds_after = preds_after.astype(np.uint16)
        
        preds_after = preds_after.sum(axis=0).astype(np.uint16)
        preds_before = preds_before.sum(axis=0).astype(np.uint16)

        preds_before[(preds_before > 0) & (preds_before < threshold) ] = 128
        preds_before[preds_before >= threshold ] = 255

        preds_before = preds_before.astype(np.uint8)

        preds_after[(preds_after > 0) & (preds_after < threshold) ] = 128
        preds_after[preds_after >= threshold ] = 255
        preds_after = preds_after.astype(np.uint8)

        preds_before = self.morph(preds_before)
        preds_after = self.morph(preds_after)
        
        mask_zero = (preds_before == 0) & (preds_after == 0)
        
        diff_deforest = (preds_after > preds_before).astype(np.uint8)

        # diff_deforest[ diff_deforest == 1] = 255
        # diff_deforest[ diff_deforest == 0] = 128
        # diff_deforest[ mask_zero] = 0

        return diff_deforest


    def morph(self, pred):
        from skimage.morphology import opening, square, closing, erosion, dilation

        mask = pred > 0
        pred [ pred < 255] = 0
        pred [ pred == 255] = 1

        #pred_opening = opening(pred, square(7))
        #pred_closing = closing(pred, square(5))
        pred = erosion(pred, square(5))
        pred = dilation(pred, square(7))

        pred[ pred == 0 ] = 128
        pred[ pred == 1 ] = 255
        pred[ ~ mask ] = 0

        # pred_closing[ pred_closing == 0 ] = 128
        # pred_closing[ pred_closing == 1 ] = 255
        # pred_closing[ ~ mask ] = 0

        # imsave('pred1.png', pred)
        # imsave('pred_open.png', pred_opening)
        # imsave('pred_close.png', pred_closing)

        return pred
        #return pred_closing

        #skimage.morphology.area_closing(image, area_threshold=64, connectivity=1, parent=None, tree_traverser=None


       

       



