from matplotlib import patches
import numpy as np
import rasterio
from rasterio.windows import Window


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
        #keep = keep[None, :, :, :]
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



def save2file_deb(b_before, transforms, meta, global_shp = None):

    import geopandas
    from geopandas.tools import overlay

    assert b_before.shape == img.shape
    if not len(b_before.shape) == 3:
        b_before = np.expand_dims(b_before, 0)
        img = np.expand_dims(img, 0)

    b_before = b_before.astype(np.uint8)
    img = img.astype(np.uint8)
    
    diff_deforest = (img > b_before).astype(np.uint8)

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


def check_overlap(w1, w2, threshold_overp = 0.95):

    w1 = ((w1 > 0)).astype(np.uint8)
    w2 = ((w2 > 0)).astype(np.uint8)
    #mask_over_images = ((w1 == 1) & (w2 == 1)).astype(np.uint8)
    mask_over_images = (w1  & w2).astype(np.uint8)

    overlR_images = mask_over_images.sum() / (w1.shape[0] * w1.shape[1])

    # should always be 1
    if (overlR_images < threshold_overp):
        return False

    return True

def decompose_image(image, patch_size = 512, offset = (0,0), write_patches = False):
        
        offset_col_init, offset_row_init = offset
        patches = []

        with rasterio.open(image) as src:

            meta = src.meta
            nb_rows = np.ceil((src.height - offset_row_init) / patch_size).astype(int)
            nb_cols = np.ceil((src.width - offset_col_init) / patch_size).astype(int)
            
            for row in range(nb_rows):

                offset_row = offset_row_init + row * patch_size
                for col in range(nb_cols):

                    offset_col = offset_col_init  + col * patch_size                    
                    block_width = min(patch_size, src.width - offset_col)
                    block_height = min(patch_size, src.height - offset_row)

                    if ((block_width < patch_size) or  (block_height < patch_size)) :
                        continue

                    window_sized = np.zeros((3, patch_size, patch_size), dtype=np.uint8)
                    crop_window = Window(offset_col, offset_row, block_width, block_height)
                    w = src.read(window=crop_window)
                    
                    window_sized[ ... , :w.shape[1], :w.shape[2]] = w
                    
                    if not check_overlap(w[0, ...], w[1, ...]):
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

                    #patches.append([window_sized, transform_window, base_fname])
                    patches.append([window_sized[0:2, ...], window_sized[2:, ...], transform_window, crop_window])

        return patches


