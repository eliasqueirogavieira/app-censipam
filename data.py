import glob
from os import path
from xml.etree.ElementPath import prepare_descendant
import numpy as np
from skimage.exposure import match_histograms
from functools import partial

from skimage.io import imread
import rasterio
import os
from os import makedirs
from genericpath import exists
from rasterio.windows import Window
from pathlib import Path


def remove_files(pathdir):
    files = glob.glob(pathdir + '*')
    for f in files:
        os.remove(f)

def create_dir(folder):
    
    if exists(folder):
        makedirs(folder)
    else:
        makedirs(folder)


def check_overlap(w1, w2, threshold_overp = 0.95):

    w1 = ((w1 > 0)).astype(np.uint8)
    w2 = ((w2 > 0)).astype(np.uint8)
    #mask_over_images = ((w1 == 1) & (w2 == 1)).astype(np.uint8)
    mask_over_images = (w1 & w2).astype(np.uint8)
    overlR_images = mask_over_images.sum() / (w1.shape[0] * w1.shape[1])

    # should always be 1
    if (overlR_images < threshold_overp):
        return False

    return True


class geo_patches(object):

    def __init__(self, patch_size, tif_metadata, nb_channels = 2):

        self.patch_size = patch_size
        self.nb_channels = nb_channels
        self.meta = tif_metadata
        self.patches = []
        self.transforms = []
        self.base_fnames = []
        self.crop_windows = []

    def add_patch(self, patch, transform, base_fname, crop_window):
        assert patch.shape[1] == self.patch_size[0]
        assert patch.shape[2] == self.patch_size[1]
        
        self.patches.append(patch)
        self.transforms.append(transform)
        self.base_fnames.append(base_fname)
        self.crop_windows.append(crop_window)

    def get_patch(self, index_pos):
        assert index_pos < len(self.patches) 
        return (self.patches[index_pos], self.transforms[index_pos])

    def get_transform(self, index_pos):
        assert index_pos < len(self.patches) 
        return self.transforms[index_pos]

    def get_patch_only(self, index_pos):
        assert index_pos < len(self.patches) 
        return self.patches[index_pos]

    def get_patch_list(self):
        return self.patches

    def normalize_image(patches, type = np.uint32):

        patches = patches.astype(np.float32) / (2. ** 16 - 1)
        patches = 1



class iceye:

    @staticmethod
    def __preprocess_chain(patch, normalize=True, histogram_match = True):

        patch = np.transpose(patch, (1, 2, 0))
        if normalize:
            patch = patch.astype(np.float32) / (2.**16 - 1.)

        if histogram_match:
            patch[..., 0] = match_histograms(patch[..., 0], patch[..., 1])

        patch = np.stack([patch[..., 0], patch[..., 1], np.zeros((patch.shape[0], patch.shape[1]), dtype=patch.dtype)], axis=2)
        patch = np.transpose(patch, (2,0, 1))
        return patch


    @staticmethod
    def apply_preprocessing_chain(patches):

        func = partial(iceye.__preprocess_chain, normalize = True, histogram_match = True)
        patches = list(map(func, patches))
        return patches


    @staticmethod
    def clip_percentil(filename):

        with rasterio.open(filename) as src0:
            meta = src0.meta
            meta['dtype'] = 'uint16'

        im = imread(filename)
        perc_0 = np.percentile(im, q=99.)
        im = np.clip(im, a_min=0, a_max=perc_0)

        if im[0, ...].max() > 0.0:
            im[0, ...] = im[0, ...] / im[0, ...].max()
        
        if im[1, ...].max() > 0.0:
            im[1, ...] = im[1, ...] / im[1, ...].max()

        im = np.round(im * (2**16-1))
        im = im.astype(np.uint16)

        with rasterio.open(filename, 'w', **meta) as dst:
            dst.write_band(1, im[0, ...])
            dst.write_band(2, im[1, ...])

    @staticmethod
    def pre_process(filename_stack, cfg):

        iceye.clip_percentil(filename_stack)
        geo_patches = iceye.decompose_image(filename_stack)
        geo_patches.patches =  iceye.apply_preprocessing_chain(geo_patches.patches)
        return geo_patches

    @staticmethod
    def decompose_image(image, patch_size = 512, write_patches = False):

        with rasterio.open(image) as src:

            meta = src.meta
            nb_rows = np.ceil(src.height / patch_size).astype(int)
            nb_cols = np.ceil(src.width / patch_size).astype(int)
            patches = geo_patches(patch_size, meta, nb_channels = 2)
            offset_col = 0
            offset_row = 0

            for row in range(nb_rows):
                offset_row = row * patch_size
                
                for col in range(nb_cols):
                    offset_col = col * patch_size
                    block_width = min(patch_size, src.width - offset_col)
                    block_height = min(patch_size, src.height - offset_row)

                    if ((block_width < patch_size) or (block_height < patch_size)) :
                        continue

                    window_sized = np.zeros((2, patch_size, patch_size), dtype=np.uint16)
                    crop_window = Window(offset_col, offset_row, block_width, block_height)
                    w = src.read(window=crop_window)
                    
                    window_sized[ ... , :w.shape[1], :w.shape[2]] = w
                    
                    if not check_overlap(w[0, ...], w[1, ...]):
                        continue

                    transform_window = src.window_transform(crop_window)

                    fileout = "file_{}_{}.tif".format(row, col)
                    meta['width'] = block_width
                    meta['height'] = block_height
                    meta['transform'] = transform_window

                    if write_patches:
                        with rasterio.open(fileout, 'w', **meta) as dst:
                            dst.write_band(1, w[0, ...])
                            dst.write_band(2, w[1, ...])

                    patches.add_patch(window_sized, transform_window)

        return patches


class sentinel:

    @staticmethod
    def processBand(band, bitdepth = 8):

        mask = band > 0
        valdata = band [mask]
        
        pinf = np.percentile(valdata, 0.001)
        psup = np.percentile(valdata, 99.9)
        tmp = np.clip(band, a_min=pinf, a_max=psup)

        band[mask] = tmp[mask]
        max_val = band.max()
        band = band / max_val

        tmp = np.clip(band, a_min=1./(2**bitdepth-1), a_max=1.0)
        band[mask] = tmp[mask]

        band = (band * (2**bitdepth - 1)).astype(np.uint8)
        return band

    @staticmethod
    def __set_nodata_value(in_filename, out_filename):
        with rasterio.open(in_filename, 'r+') as source:
            if source.meta['nodata'] == 0: return
            band1 = source.read(1)   
            band2 = source.read(2)   
            band3 = source.read(3)   
            band4 = source.read(4)   

        meta = source.meta
        meta['nodata'] = 0
        meta['count'] = 4

        with rasterio.open(out_filename, 'w', **meta) as dst:
            dst.write_band(1, band1)
            dst.write_band(2, band2)
            dst.write_band(3, band3)
            dst.write_band(4, band4)

    @staticmethod
    def __process_to8bit(in_filename, out_filename):

        with rasterio.open(in_filename, 'r+') as source:
            if source.meta['dtype'] == 'uint8': return
            band1 = source.read(1)
            band2 = source.read(2)   
            band3 = source.read(3)   
            band4 = source.read(4)   
            
        band1 = sentinel.processBand(band1)
        band2 = sentinel.processBand(band2)
        band3 = sentinel.processBand(band3)
        band4 = sentinel.processBand(band4)

        meta = source.meta
        meta['dtype'] = 'uint8'
        meta['count'] = 4
        meta['nodata'] = 0

        with rasterio.open(out_filename, 'w', **meta) as dst:
             dst.write_band(1, band1)
             dst.write_band(2, band2)
             dst.write_band(3, band3)
             dst.write_band(4, band4)

    @staticmethod
    def pre_process(fname, config):

        sentinel.__set_nodata_value(fname, fname)
        sentinel.__process_to8bit(fname, fname)

    
    @staticmethod
    def decompose_image(image, patch_size = 4096, offset = (0,0), write_patches = False):
        
        offset_col_init, offset_row_init = offset

        with rasterio.open(image) as src:

            meta = src.meta
            nb_rows = np.ceil((src.height - offset_row_init) / patch_size).astype(int)
            nb_cols = np.ceil((src.width - offset_col_init) / patch_size).astype(int)
            patches = geo_patches((patch_size, patch_size), meta, nb_channels = 4)

            for row in range(nb_rows):

                offset_row = offset_row_init + row * patch_size
                for col in range(nb_cols):

                    offset_col = offset_col_init  + col * patch_size                    
                    block_width = min(patch_size, src.width - offset_col)
                    block_height = min(patch_size, src.height - offset_row)

                    window_sized = np.zeros((4, patch_size, patch_size), dtype=np.uint8)
                    crop_window = Window(offset_col, offset_row, block_width, block_height)
                    w = src.read(window=crop_window)
                    
                    window_sized[ ... , :w.shape[1], :w.shape[2]] = w
                    
                    if not check_overlap(w[0, ...], w[3, ...]):
                        continue

                    transform_window = src.window_transform(crop_window)

                    base_fname = f"crop_{row}_{col}"

                    fileout = "file_before_{}_{}.tif".format(row, col)
                    meta['width'] = block_width
                    meta['height'] = block_height
                    meta['transform'] = transform_window
                    meta['count'] = 2

                    if write_patches:
                        with rasterio.open(fileout, 'w', **meta) as dst:
                            dst.write_band(1, w[0, ...])
                            dst.write_band(2, w[1, ...])
                    
                    fileout = "file_after_{}_{}.tif".format(row, col)
                    if write_patches:
                        with rasterio.open(fileout, 'w', **meta) as dst:
                            dst.write_band(1, w[2, ...])
                            dst.write_band(2, w[3, ...])

                    patches.add_patch(window_sized, transform_window, base_fname, crop_window)

        return patches



def pre_process(fname, config, patch_size = 512, offset = (0,0)):

    data_preprocess = eval(config.IN_DATA['format'])
    data_preprocess.pre_process(fname, config)

    geo_patches = data_preprocess.decompose_image(fname, patch_size = patch_size, 
                                                    offset = offset, write_patches=False)
    return geo_patches