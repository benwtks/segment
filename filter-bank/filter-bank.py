import numpy as np
import cudf
from cuml import KMeans
import cupy as cp
from cupyx.scipy.ndimage import gaussian_filter
from cupy.fft import fft2, ifft2, fftshift
from cupy.random import normal
from skimage import data, io
from skimage.exposure import equalize_hist
from skimage.filters import gabor_kernel
from scipy.stats import norm # not available in cupyx.scipy
from math import log2, sqrt
import itertools
from random import randint, sample

def gaussian(kernlen=21, nsig=3):
    linspace = np.linspace(-nsig, nsig, kernlen+1)
    kern = cp.outer(*(cp.diff(cp.asarray(norm.cdf(linspace))),)*2)
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
        kernel = cp.real(cp.asarray(gabor_kernel(frequency, theta=theta,sigma_x=sigma, sigma_y=sigma)))

        padding = (image.shape[0] - kernel.shape[0], image.shape[1] - kernel.shape[1]) 
        k = cp.pad(kernel, (((padding[0]+1)//2, padding[0]//2), ((padding[1]+1)//2, padding[1]//2)), 'constant')
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

def generate_feature_vectors(image, M, alpha):
  image = cp.asarray(equalize_hist(image))
  feature_images = generate_feature_images(image - image.mean())
  features = cp.zeros((image.shape[0], image.shape[1],len(feature_images)))
  nonlin = lambda i : abs((1- np.exp(-2*alpha*i))/(1+ np.exp(-2*alpha*i)))
  gaussian_kern = gaussian(M, M*0.08)

  for i in range(len(feature_images)):
    nonlin_convolve = cp.pad(nonlin(feature_images[i][1]), M//2, mode='symmetric')

    # Make a feature image for that convolved image
    for y in range(image.shape[0]):
      for x in range(image.shape[1]):
        window = nonlin_convolve[cp.ix_(range(y,y+M),range(x,x+M))]
        blurred = gaussian_filter(window, 3 * np.std(window))
        weighted = cp.einsum('ij,ij->ij', gaussian(M, 0.5*M/feature_images[i][0]), blurred)
        features[y][x][i] = cp.sum(weighted) / M**2
  return features

def filter_bank_segment(image, no_segments, M, alpha):
  features = generate_feature_vectors(image, M, alpha)
  for i in range(len(features)):
      features[i] = (features[i] - features[i].mean()) / features[i].std(axis=0)

  kmeans = KMeans(n_clusters=no_segments).fit(features.reshape((image.shape[0]*image.shape[1],features.shape[2])))
  segmented = np.zeros(image.shape)
  for i in range(image.shape[0] * image.shape[1]):
    y=i//image.shape[1]
    x=i-y*image.shape[1]
    segmented[y][x] = kmeans.labels_[i]*255/no_segments
  
  return segmented

image = io.imread('input.png', as_gray=True)
no_segments = 2
M = 20
alpha = 0.25

io.imsave("segmentation.png", filter_bank_segment(image, no_segments, M, alpha))
