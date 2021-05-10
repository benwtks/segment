import numpy as np
from skimage import data, io
from skimage.color import rgb2gray
from skimage.exposure import equalize_hist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import sys

image = io.imread(sys.argv[1], as_gray=True)

# SAMPLE is what is taken in each region/window
# neighbourhood_size is the size of each sample (n)
# region/window is omega_s
# w (region/window size) = 2n - 1
neighbourhood_size = int(sys.argv[2]) # make sure this is odd, 7 recommended
no_samples = neighbourhood_size ** 2
window_size = 2 * neighbourhood_size - 1
padded = np.pad(image, window_size // 2, mode='symmetric')
features = np.zeros((image.shape[0] * image.shape[1], (neighbourhood_size**2 +1) // 2))

#io.imsave("padded.png", padded)

c = 100

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
		sigma_squared /= no_samples**2

		features[image_y*image.shape[1]+image_x][:(neighbourhood_size**2-1)//2] = alpha[:,0]
		features[image_y*image.shape[1]+image_x][(neighbourhood_size**2-1)//2] = np.sqrt(sigma_squared)

#features shape = (image.shape[0] * image.shape[1], (neighbourhood_size**2 +1) // 2)
histogram_window_size = int(sys.argv[3]) #b, must be even because of padding
feature_images = np.zeros((features.shape[1],image.shape[0]+histogram_window_size,image.shape[1]+histogram_window_size))
for i in range(features.shape[1]):
	feature_image = np.zeros(image.shape)
	for y in range(image.shape[0]):
		for x in range(image.shape[1]):
			feature_image[y][x] = features[y*image.shape[1]+x][i]
	feature_images[i] = np.pad(feature_image, histogram_window_size // 2, mode='symmetric')
	#io.imsave(f"feature_images/feature_{i}.png", feature_images[i])

# Calculate LPH
no_bins = int(sys.argv[4])
feature_vectors = np.zeros((image.shape[0] * image.shape[1],feature_images.shape[0]*no_bins))
for y in range(image.shape[0]):
	for x in range(image.shape[1]):
		feature_vector = np.zeros((feature_images.shape[0]*no_bins,))
		for i in range(feature_images.shape[0]):
			region = feature_images[i][np.ix_(range(y,y+histogram_window_size),range(x,x+histogram_window_size))]
			feature_vector[i*no_bins:(i+1)*no_bins] = np.histogram(region,no_bins)[0]
		feature_vectors[y*image.shape[1]+x] = feature_vector

# Apply PCA
no_dimensions = 25 # fixed  as in paper
reduced_features = PCA(no_dimensions).fit(features).transform(features)

# Cluster
n_clusters= int(sys.argv[5])
kmeans = KMeans(n_clusters).fit(reduced_features)
segmented = np.zeros(image.shape)
for y in range(image.shape[0]):
	for x in range(image.shape[1]):
		label = kmeans.labels_[y*image.shape[1]+x]
		segmented[y][x] = label * 255/n_clusters

io.imsave("output.png", segmented)
