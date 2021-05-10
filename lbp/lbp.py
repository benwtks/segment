import numpy as np
from sklearn.cluster import KMeans
from skimage import data, io
from skimage.feature import local_binary_pattern
from skimage.exposure import equalize_hist
import sys

def lbp_hist_feature(image, radius_linspace, no_bins):
	lbp = lambda i, r: np.asarray(local_binary_pattern(i, 8 * r, r, 'uniform'))
	hist = np.zeros((len(radius_linspace) * no_bins))
	for i in range(len(radius_linspace)):
		hist[(i*no_bins):(i*no_bins)+no_bins] = np.histogram(lbp(image, radius_linspace[i]), bins=np.linspace(0,(radius_linspace[i]*8)+1,no_bins+1))[0]

	return hist

def segmentation(image, window_size, no_segments, radius_linspace, no_bins):
	image = equalize_hist(image)
	padded = np.pad(image - image.mean(), window_size // 2, mode='symmetric')
	segmented = np.zeros(image.shape)
	histograms = np.zeros((image.shape[0]*image.shape[1],len(radius_linspace)*no_bins))
	for y in range(image.shape[0]):
		for x in range(image.shape[1]):
			window = padded[np.ix_(range(y,y+window_size),range(x,x+window_size))]
			window = window - window.mean()
			histograms[y*image.shape[1] + x] = lbp_hist_feature(window, radius_linspace, no_bins)

	kmeans = KMeans(no_segments).fit(histograms)

	for y in range(image.shape[0]):
		for x in range(image.shape[1]):
			label = kmeans.labels_[y*image.shape[1]+x]
			segmented[y][x] = label*255/no_segments

	return segmented

image = io.imread(sys.argv[1], as_gray=True)
window_size = int(sys.argv[2]) # window size between 30-50
radii = eval(sys.argv[3]) #[1,1.25,1.5,2,3]
bins = int(sys.argv[4])
segments = int(sys.argv[5])

io.imsave("output.png", segmentation(image, window_size, segments, radii, bins))
