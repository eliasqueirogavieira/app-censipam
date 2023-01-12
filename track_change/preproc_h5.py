import numpy as np
import skimage.io
import matplotlib.pyplot as plt


im = skimage.io.imread("ICEYE_X2_SLC_SM_57947_20210617T133121.tif")

im_before = im[0, ...]
im_after = im[1, ...]

#im_before = im_after

# log_image = 1/(1 +  np.exp(-im_before))
# plt.imshow(log_image)


# hist_0, edges_0 = np.histogram(log_image, bins=100)
# centers_0 = [0.5*(edges_0[i] + edges_0[i+1])  for i  in range(len(edges_0)-1)]
# plt.figure(0)
# plt.plot(centers_0, hist_0, '-r')
# plt.bar(centers_0, hist_0, width=0.001)
# plt.title('Perc. 100')
# plt.grid()


hist_0, edges_0 = np.histogram(im_before, bins=100)
centers_0 = [0.5*(edges_0[i] + edges_0[i+1])  for i  in range(len(edges_0)-1)]
plt.subplot(2,1,1)
plt.plot(centers_0, hist_0, '-r')
plt.bar(centers_0, hist_0, width=0.001)
plt.title('Perc. 100')
plt.grid()

perc_0 = np.percentile(im_before, q=98)
edges_0 = np.linspace(0, perc_0, 21)
centers_ = [0.5*(edges_0[i] + edges_0[i+1])  for i  in range(len(edges_0)-1)]
hist_0, edges_0 = np.histogram(im_before, edges_0)
centers_0 = [0.5*(edges_0[i] + edges_0[i+1])  for i  in range(len(edges_0)-1)]
plt.subplot(2,1,2)
plt.plot(centers_0, hist_0, '-r')
plt.bar(centers_0, hist_0, width=0.001)
plt.title('Perc. 98')
plt.grid()


index_large = im_before > perc_0
print("# of samples > percentil 98: ".format( np.sum(index_large) ))
im_before_clip = np.clip(im_before, a_min=0, a_max=perc_0)




max_idx = np.unravel_index(im_before.argmax(), im_before.shape)

perc = np.linspace(0, 100, 50)
c = [np.percentile(im_before, v) for v in perc ]
plt.figure(2)
plt.plot(c, perc, '-r.')
plt.show()
plt.grid()







print("End")