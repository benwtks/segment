import numpy as np
from skimage import data, io
from skimage.color import rgb2gray
from skimage.exposure import equalize_hist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import sys

image = io.imread('~/image_5.png', as_gray=True)

# SAMPLE is what is taken in each region/window
# neighbourhood_size is the size of each sample (n)
# region/window is omega_s
# w (region/window size) = 2n - 1
neighbourhood_size = 7 # make sure this is odd
no_samples = neighbourhood_size ** 2
window_size = 2 * neighbourhood_size - 1
padded = np.pad(image, window_size // 2, mode='symmetric')
features = np.zeros((image.shape[0] * image.shape[1], (neighbourhood_size**2 +1) // 2))

#io.imsave("padded.png", padded)

c = 100

# we have a ohm_s at each pixel
for image_y in range(image.shape[0]):
	for image_x in range(image.shape[1]):
		print(image_x,image_y)
		region = padded[np.ix_(range(image_y,image_y+window_size),range(image_x,image_x+window_size))]

		Y_bar_t = np.zeros((no_samples, (neighbourhood_size**2 - 1)//2)) # neighbour values
		y = np.zeros((no_samples, 1)) # centre values
		for i in range(window_size-neighbourhood_size+1):
			for j in range(window_size-neighbourhood_size+1):
				sample = region[np.ix_(range(i,i+neighbourhood_size), range(j,j+neighbourhood_size))]
				sample = sample - sample.mean()
				sample = sample.flatten()
				y[i*(window_size-neighbourhood_size+1)+j][0] = sample[neighbourhood_size**2 // 2]
				Y_bar_t[i*(window_size-neighbourhood_size+1)+j] = sample[:(neighbourhood_size**2-1)//2] + sample[(neighbourhood_size**2-1)//2 + 1 :]

		first_part = np.zeros(((neighbourhood_size**2 - 1) // 2, (neighbourhood_size**2 - 1)//2))
		for s in range(no_samples):
			first_part += np.matmul(Y_bar_t[s][np.newaxis,:].T, Y_bar_t[s][np.newaxis,:])
		first_part += c**2 * np.identity((neighbourhood_size**2-1)//2)
		first_part = np.linalg.pinv(first_part)

		second_part = np.matmul(Y_bar_t.T, y).sum(axis=1)[np.newaxis,:].T

		alpha = np.matmul(first_part, second_part)

		sigma_squared = 0
		for s in range(no_samples):
			sigma_squared += (y[s][0] - np.matmul(alpha.T, Y_bar_t[s][np.newaxis,:].T)[0][0])**2
		sigma_squared /= window_size**2

		features[image_y*image.shape[1]+image_x][:(neighbourhood_size**2-1)//2] = alpha[:,0]
		features[image_y*image.shape[1]+image_x][(neighbourhood_size**2-1)//2] = np.sqrt(sigma_squared)

#features shape = (image.shape[0] * image.shape[1], (neighbourhood_size**2 +1) // 2)
feature_images = np.zeros((features.shape[1],image.shape[0],image.shape[1]))

for i in range(features.shape[1]):
	for y in range(image.shape[0]):
		for x in range(image.shape[1]):
			feature_images[i][y][x] = features[y*image.shape[1]+x][i]

# Calculate LPH
histogram_window_size = 20 #b
no_bins = 10
no_histogram_windows = (image.shape[0] // histogram_window_size) * (image.shape[1] // histogram_window_size)
feature_vectors = np.zeros((no_histogram_windows,feature_images.shape[0]*no_bins))
index = 0
for y in range(0,image.shape[0]-histogram_window_size+1, histogram_window_size):
	for x in range(0,image.shape[1]-histogram_window_size+1, histogram_window_size):
		feature_vector = np.zeros((feature_images.shape[0]*no_bins,))
		for i in range(feature_images.shape[0]):
			region = feature_images[i][np.ix_(range(y,y+histogram_window_size),range(x,x+histogram_window_size))]
			feature_vector[i*no_bins:(i+1)*no_bins] = np.histogram(region,no_bins)[0]
		feature_vectors[index] = feature_vector
		index += 1


# Apply PCA
no_dimensions = 25
reduced_features = PCA(no_dimensions).fit(features).transform(features)

# Cluster
n_clusters=4
kmeans = KMeans(n_clusters).fit(reduced_features)
segmented = np.zeros(image.shape)
index = 0
for y in range(0,image.shape[0] - histogram_window_size + 1, histogram_window_size):
	for x in range(0,image.shape[1] - histogram_window_size + 1, histogram_window_size):
		label = kmeans.labels_[index]
		ix = np.ix_(range(y,y+histogram_window_size),range(x,x+histogram_window_size))
		segmented[ix] = label*255/n_clusters
		index += 1
		
io.imsave("segmented.png", segmented)
