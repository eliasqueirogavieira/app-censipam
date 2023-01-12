import numpy as np
import skimage.io
import matplotlib.pyplot as plt


im = skimage.io.imread("ICEYE_X2_SLC_SM_57947_20210617T133121.tif")

im_before = im[0, ...]
im_after = im[1, ...]

diff = im_after - im_before


perc = np.linspace(0, 100, 50)
c = [np.percentile(diff, v) for v in perc ]

plt.plot(c, perc, '-r.')
plt.show()
plt.grid()

hist_0, edges_0 = np.histogram(diff, bins=202)
centers_0 = [0.5*(edges_0[i] + edges_0[i+1])  for i  in range(len(edges_0)-1)]

plt.figure(2)

plt.plot(centers_0, hist_0, '-r')
plt.bar(centers_0, hist_0, width=0.001)
plt.title('Perc. 100')
plt.grid()




# plt.plot(hist_0)

# plt.figure(0)
# plt.imshow(im_before)

# plt.figure(1)
# plt.imshow(im_after)

print("End")