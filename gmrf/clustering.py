import numpy as np
from cuml import KMeans, PCA
import cudf
from skimage import data, io
from skimage.color import rgb2gray, gray2rgb
# difference between np and cp, np throws error if ix out of range. Be aware of this.

n_clusters = 2
image = io.imread('input2-crop.png', as_gray=True)
shape = image.shape

# Apply PCA
no_dimensions = 25
features = np.load('features.npy')
reduced_features = PCA(n_components=no_dimensions).fit(features).transform(features)

# Segmentation
# localisation issues but this is how the paper does it... Maybe should change other algorithms to match? idkkk
histogram_window_size = 20
no_histogram_windows = (shape[0] // histogram_window_size) * (shape[1] // histogram_window_size)
kmeans = KMeans(n_clusters=n_clusters).fit(reduced_features)
segmented = np.zeros(shape)
index = 0
for y in range(0,shape[0] - histogram_window_size + 1, histogram_window_size):
	for x in range(0,shape[1] - histogram_window_size + 1, histogram_window_size):
		label = kmeans.labels_[index]
		ix = np.ix_(range(y,y+histogram_window_size),range(x,x+histogram_window_size))
		segmented[ix] = label*255/n_clusters
		index += 1
		
io.imsave("segmented.png", segmented)
