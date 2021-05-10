import numpy as np
from numpy.random import normal
from skimage import data, io
from skimage.color import rgb2gray, gray2rgb
from skimage.feature import local_binary_pattern
from skimage.exposure import equalize_hist
from sklearn.cluster import KMeans
from skimage.filters import gabor_kernel
from scipy.stats import norm
from scipy.ndimage import gaussian_filter, convolve
from scipy.fft import fft2, ifft2, fftshift
from math import log2, sqrt
import itertools
from random import randint, sample
import sys

def gaussian(kernlen=21, nsig=3):
    linspace = np.linspace(-nsig, nsig, kernlen+1)
    kern = np.outer(*(np.diff(norm.cdf(linspace)),)*2)
    return kern/kern.sum()

def generate_feature_images(image):
  image_DFT = fft2(image)
  convolutions = []
  energies = []
  total_energy = 0
  i = 0
  for theta in np.arange(0, 7/6,step=1/6) * np.pi:
    for sigma in (1, 3):
      for frequency in 2**np.arange(3,int(log2(image.shape[0]))-1,step=1) * sqrt(2):
        kernel = np.real(gabor_kernel(frequency, theta=theta,sigma_x=sigma, sigma_y=sigma))

        padding = (image.shape[0] - kernel.shape[0], image.shape[1] - kernel.shape[1]) 
        k = np.pad(kernel, (((padding[0]+1)//2, padding[0]//2), ((padding[1]+1)//2, padding[1]//2)), 'constant')
        convolution = image_DFT * fft2(fftshift(k))
        convolutions.append((frequency, convolution))

        energy = 0
        for u in range(convolution.shape[0]):
          for v in range(convolution.shape[1]):
            energy += abs(convolution[u][v]) ** 2

        total_energy += energy
        energies.append({'id':i, 'energy':energy})
        i += 1

  sorted_energies = sorted(energies, key=(lambda i: i['energy']),reverse=True)
  R_squared = 0
  target = 0.95 * total_energy
  feature_images = []
  for i in range(len(sorted_energies)):
    if R_squared >= target:
      print("Using", i, "Gabor filters")
      break

    R_squared += sorted_energies[i]['energy']
    selected_kernel = convolutions[sorted_energies[i]['id']]
    feature_images.append((selected_kernel[0], np.real(ifft2(selected_kernel[1]))))

  return feature_images

def generate_lbp_features(image, M, alpha, radius, no_bins):
  image = equalize_hist(image)
  feature_images = generate_feature_images(image - image.mean())
  nonlin = lambda i : abs((1- np.exp(-2*alpha*i))/(1+ np.exp(-2*alpha*i)))

  features = np.zeros((image.shape[0] * image.shape[1],len(feature_images)*no_bins))
  convolved = [np.pad(nonlin(feature_images[i][1]), M//2, mode='symmetric') for i in range(len(feature_images))]
  convolved = [c - c.mean() for c in convolved]

  lbp = lambda i, r: local_binary_pattern(i, 8 * r, r, 'uniform')

  for y in range(image.shape[0]):
    for x in range(image.shape[1]):
      for i in range(len(feature_images)):
        window = convolved[i][np.ix_(range(y,y+M),range(x,x+M))]
        features[y*image.shape[0]+x][i*no_bins:(i+1)*no_bins] = np.histogram(
            lbp(window, radius), bins=np.linspace(0,(radius*8)+1,no_bins+1))[0]
  return features

def lbp_filter_bank_segment(image, no_segments, M, alpha, radius, no_bins):
  features = generate_lbp_features(image, M, alpha, radius, no_bins)

  kmeans = KMeans(no_segments).fit(features)
  segmented = np.zeros(image.shape)
  for i in range(image.shape[0] * image.shape[1]):
    y=i//image.shape[1]
    x=i-y*image.shape[1]
    segmented[y][x] = 50+kmeans.labels_[i]*100/no_segments
  
  return segmented

image = io.imread(sys.argv[1], as_gray=True)
window_size = int(sys.argv[2])
alpha = 0.25 #fixed as paper suggests
radius = int(sys.argv[3])
no_bins = int(sys.argv[4])
no_segments = int(sys.argv[5])

io.imsave("output.png", lbp_filter_bank_segment(image, no_segments, window_size, alpha, radius, no_bins))
