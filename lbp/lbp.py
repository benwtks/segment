import numpy as np
from sklearn.cluster import KMeans
from skimage import data, io
from skimage.feature import local_binary_pattern
from skimage.exposure import equalize_hist
from sklearn.decomposition import PCA
import sys

def lbp_hist_feature(image, radius_linspace, total_no_bins):
	lbp = lambda i, r: np.asarray(local_binary_pattern(i, 8 * r, r, 'uniform'))
	hist = np.zeros((total_no_bins,))
	cumulative_bin_total = 0
	for i in range(1,max_radius+1):
		hist[cumulative_bin_total:cumulative_bin_total + i * 8 + 1] = np.histogram(lbp(image, i), bins=np.linspace(0,i*8+1,i*8+2))[0]
		cumulative_bin_total += i * 8 + 1

	return hist

def segmentation(image, window_size, no_segments, max_radius):
	image = equalize_hist(image)
	padded = np.pad(image - image.mean(), window_size // 2, mode='symmetric')
	segmented = np.zeros(image.shape)
	total_no_bins = max_radius * (4 * max_radius + 5)
	histograms = np.zeros((image.shape[0]*image.shape[1],total_no_bins))
	for y in range(image.shape[0]):
		for x in range(image.shape[1]):
			window = padded[np.ix_(range(y,y+window_size),range(x,x+window_size))]
			window = window - window.mean()
			histograms[y*image.shape[1] + x] = lbp_hist_feature(window, max_radius, total_no_bins)

	features = PCA(25).fit(histograms).transform(histograms)
	kmeans = KMeans(no_segments).fit(features)

	for y in range(image.shape[0]):
		for x in range(image.shape[1]):
			label = kmeans.labels_[y*image.shape[1]+x]
			segmented[y][x] = label*255/no_segments

	return segmented

image = io.imread(sys.argv[1], as_gray=True)
window_size = int(sys.argv[2]) # window size between 30-50
max_radius = int(sys.argv[3])
segments = int(sys.argv[4])

io.imsave("output.png", segmentation(image, window_size, segments, max_radius))
