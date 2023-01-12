import numpy as np
import skimage.io
import matplotlib.pyplot as plt

from skimage import exposure
from skimage.exposure import match_histograms
import rasterio


def stack_with_rasterio(img, outfile):
    
    with rasterio.open("ICEYE_X2_SLC_SM_57947_20210617T133121.tif") as src0:
        meta = src0.meta
    
    #meta.update(count = len(img_list))

    with rasterio.open(outfile, 'w', **meta) as dst:
        #for id, layer in enumerate(img_list, start=1):
        #    with rasterio.open(layer) as src1:
        #        dst.write_band(id, src1.read(1))
        
                dst.write_band(1, img[0, ...])
                dst.write_band(2, img[1, ...])


im = skimage.io.imread("ICEYE_X2_SLC_SM_57947_20210617T133121.tif")


# perc_0 = np.percentile(im, q=99)
# index_large = im > perc_0
# print("# of samples > percentil 99: ".format( np.sum(index_large) ))
# im = np.clip(im, a_min=0, a_max=perc_0)

# stack_with_rasterio(im, 'stack_0.tif')

#max_idx = np.unravel_index(im.argmax(), im.shape)

# perc = np.linspace(0, 100, 50)
# c = [np.percentile(im_clip[0, ...], v) for v in perc ]
# plt.figure()
# plt.plot(c, perc, '-r.')

# perc = np.linspace(0, 100, 50)
# c = [np.percentile(im_clip[1, ...], v) for v in perc ]
# plt.plot(c, perc, '-b.')

# plt.show()
# plt.grid()


# plt.figure()
# plt.subplot(1,2,1)
# plt.imshow(im_clip[0, ...], cmap='gray')

# plt.subplot(1,2,2)
# plt.imshow(im_clip[1, ...], cmap='gray')


tmp = np.linalg.lstsq(im[0, ...], im[1, ...])


im[0, ...] = match_histograms(im[0, ...], im[1, ...])

stack_with_rasterio(im, 'stack.tif')


perc = np.linspace(0, 100, 50)
c = [np.percentile(im[0, ...], v) for v in perc ]
plt.figure()
plt.plot(c, perc, '-r.')

perc = np.linspace(0, 100, 50)
c = [np.percentile(im[1, ...], v) for v in perc ]
plt.plot(c, perc, '-b.')


plt.figure()
plt.subplot(1,2,1)
plt.imshow(im[0, ...], cmap='gray')

plt.subplot(1,2,2)
plt.imshow(im[1, ...], cmap='gray')


print("End")