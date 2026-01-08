                                                                                   
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

from google.colab import drive
drive.mount('/content/drive')
     
from os import path
import math        
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
import sys

                                                             
datadir = "/content/drive/My Drive/image_processing/" 

utilfn = datadir + "utils"
get_ipython().system('cp -r "$utilfn" .')
samplesfn = datadir + "samples"
get_ipython().system('cp -r "$samplesfn" .')

                                                    
get_ipython().system('mkdir "images"')
get_ipython().system('mkdir "images/outputs"')

                     
import utils
from utils.io import read_image, write_image, read_hdr_image, write_hdr_image
from utils.display import display_images_linear_rescale, rescale_images_linear
from utils.hdr_helpers import gsolve
from utils.hdr_helpers import get_equirectangular_image
from utils.bilateral_filter import bilateral_filter

imdir = 'samples/my_samples'
img = cv2.imread(imdir + '/' + '120.jpeg')
print(img.shape)                    
cv2.imshow("original", img)
     
cropped_image = img[1600:1950, 2570:2920]
cv2.imshow("cropped", cropped_image)            
cv2.imwrite("Cropped Image.jpeg", cropped_image)

imdir = 'samples/my_samples'
                                                                                                                                                                                         
imfns = ['30.jpeg', '120.jpeg', '556.jpeg', '2398.jpeg', '9804.jpeg', '17241.jpeg']
exposure_times = [1/30.0, 1/120.0, 1/556.0, 1/2398.0, 1/9804.0, 1/17241.0]
ldr_images = []
for f in np.arange(len(imfns)):
    im = read_image(imdir + '/' + imfns[f])
    if f==0:
        imsize = int((im.shape[0] + im.shape[1])/2)                                  
        ldr_images = np.zeros((len(imfns), imsize, imsize, 3))
    ldr_images[f] = cv2.resize(im, (imsize, imsize))

background_image_file = imdir + '/' + 'background.jpeg'
background_image = read_image(background_image_file)


def make_hdr_naive(ldr_images: np.ndarray, exposures: list) -> (np.ndarray, np.ndarray):
    N, H, W, C = ldr_images.shape       
    assert N == len(exposures)
  
    constant = 0.1                                
    irradiances = ldr_images / np.array(exposures).reshape(-1, 1, 1, 1)
    irradiances = irradiances - np.min(irradiances)                                  
    hdr_image = np.mean(irradiances, axis=0)
                                      
    log_irradiances = np.log(irradiances + constant)

    return hdr_image, log_irradiances

def display_hdr_image(im_hdr, zero_log=False):                 
    constant = 0.1 if zero_log else 0
    img_out = rescale_images_linear(np.log(im_hdr + constant))
    plt.imshow(img_out)
    plt.show()
              
naive_hdr_image, naive_log_irradiances = make_hdr_naive(ldr_images, exposure_times)
                 
write_hdr_image(naive_hdr_image, 'images/outputs/naive_hdr.hdr')
       
print('HDR Image')
display_hdr_image(naive_hdr_image)
                                          
display_images_linear_rescale(ldr_images)                                          
display_images_linear_rescale(naive_log_irradiances)

def make_hdr_weighted(ldr_images: np.ndarray, exposure_times: list) -> (np.ndarray, np.ndarray):
    N, H, W, C = ldr_images.shape
    assert N == len(exposure_times)                                                 
    w = np.vectorize(lambda z: float(128 - np.abs(z - 128)))
    weights = w(ldr_images)
                          
    rescaled_images = ldr_images / np.array(exposure_times).reshape(-1, 1, 1, 1)                                       
    weights_sum = np.sum(weights, axis=0)
    hdr_image = np.sum(weights * rescaled_images, axis=0)                                                              
    hdr_image /= weights_sum
    
    return hdr_image
                
weighted_hdr_image = make_hdr_weighted(ldr_images, exposure_times)               
write_hdr_image(weighted_hdr_image, 'images/outputs/weighted_hdr.hdr')  
display_hdr_image(weighted_hdr_image)

log_diff_im = np.log(weighted_hdr_image)-np.log(naive_hdr_image)
print('Min ratio = ', np.exp(log_diff_im).min(), '  Max ratio = ', np.exp(log_diff_im).max())
plt.figure()
plt.imshow(rescale_images_linear(log_diff_im))

def make_hdr_estimation(ldr_images: np.ndarray, exposure_times: list, lm)-> (np.ndarray, np.ndarray):
    N, H, W, _ = ldr_images.shape
        
    assert N == len(exposure_times)
    sample_num = 1200
                                     
    xs = np.random.randint(W, size=sample_num)
    ys = np.random.randint(H, size=sample_num)                   
    w = np.vectorize(lambda z: float(128 - np.abs(z - 128)))
    log_exposure_times = np.log(exposure_times)    
    g = np.zeros((3, 256))
    Z = np.zeros((N, sample_num))
    for c in range(3):
        for i in range(sample_num):
            Z[:, i] = ldr_images[:, ys[i], xs[i], c]
        Z = Z.astype(np.int64)
        g[c], lE = gsolve(Z, log_exposure_times, lm, w)

    hdr_image = np.zeros((H, W, 3))
    log_irradiances = np.zeros((N, H, W, 3))

    for c in range(3):
        weights = w(ldr_images[..., c])
        weights /= np.max(weights)
        g_c = g[c].take(ldr_images[..., c].astype(np.int64))
        log_irradiances[...,c] = g_c - log_exposure_times[:, np.newaxis, np.newaxis]
        sum_weights = np.sum(weights, axis=0)
        hdr_image[..., c] = np.sum(weights * log_irradiances[..., c], axis=0) /N
                         
    log_irradiances = (log_irradiances - np.min(log_irradiances))/(np.max(log_irradiances) - np.min(log_irradiances))
    return hdr_image, np.exp(log_irradiances), g

lm = 50                    
calib_hdr_image, calib_log_irradiances, g = make_hdr_estimation(ldr_images, exposure_times, lm)                  
write_hdr_image(calib_hdr_image.astype('float32'), 'images/outputs/calib_hdr.hdr')        
display_hdr_image(calib_hdr_image, True)
                                                    
log_diff_im = np.log(calib_hdr_image/calib_hdr_image.mean())-np.log(weighted_hdr_image/weighted_hdr_image.mean())
print('Min ratio = ', np.exp(log_diff_im).min(), '  Max ratio = ', np.exp(log_diff_im).max())
plt.figure()
plt.imshow(rescale_images_linear(log_diff_im))
                                             
display_images_linear_rescale(ldr_images)                                             
display_images_linear_rescale(calib_log_irradiances)
                                  
N, NG = g.shape
labels = ['R', 'G', 'B']
plt.figure()
for n in range(N):
    plt.plot(g[n], range(NG), label=labels[n])
plt.gca().legend(('R', 'G', 'B'))

plt.figure()
for n in range(N):
    plt.plot(range(NG), g[n], label=labels[n])
plt.gca().legend(('R', 'G', 'B'))

def weighted_log_error(ldr_images, hdr_image, log_irradiances):                                                                                            
    N, H, W, C = ldr_images.shape
    w = 1-abs(ldr_images - 0.5)*2
    err = 0
    for n in np.arange(N):
        err += np.sqrt(np.multiply(w[n], (log_irradiances[n]-np.log(hdr_image))**2).sum()/w[n].sum())/N 
    return err
  
err = weighted_log_error(ldr_images, naive_hdr_image, naive_log_irradiances)
print('naive:  \tlog range = ', round(np.log(naive_hdr_image).max() - np.log(naive_hdr_image).min(),3), '\tavg RMS error = ', round(err,3))
err = weighted_log_error(ldr_images, weighted_hdr_image, naive_log_irradiances)
print('weighted:\tlog range = ', round(np.log(weighted_hdr_image).max() - np.log(weighted_hdr_image).min(),3), '\tavg RMS error = ', round(err,3))
err = weighted_log_error(ldr_images, calib_hdr_image, calib_log_irradiances)
print('calibrated:\tlog range = ', round(np.log(calib_hdr_image).max() - np.log(calib_hdr_image).min(),3), '\tavg RMS error = ', round(err,3))

                                                         
display_images_linear_rescale(np.log(np.stack((naive_hdr_image/naive_hdr_image.mean(), weighted_hdr_image/weighted_hdr_image.mean(), calib_hdr_image/calib_hdr_image.mean()), axis=0)))

def panoramic_transform(hdr_image):
    H, W, C = hdr_image.shape
    assert H == W
    assert C == 3
       
    N = np.zeros((H, W, C))
    R = np.zeros((H, W, C))

    lin_x = np.linspace(-1, 1, W)
    lin_y = np.linspace(-1, 1, H)
    X, Y = np.meshgrid(lin_x, lin_y)
    Z = np.sqrt(np.maximum(0, 1 - X**2 - Y**2))
    N[:, :, 0] = X
    N[:, :, 1] = Y
    N[:, :, 2] = Z

    V = np.zeros((H, W, C))
    V[..., 2] = -1
    V /= np.linalg.norm(V, axis=-1, keepdims=True)

    R = V - 2 * np.sum(V * N, axis=-1, keepdims=True) * N 
    R /= np.linalg.norm(R, axis=-1, keepdims=True)

    plt.imshow((N+1)/2)
    plt.show()
    plt.imshow((R+1)/2)
    plt.show()

    equirectangular_image = get_equirectangular_image(R, hdr_image)
    return equirectangular_image

hdr_mirrorball_image = read_hdr_image('images/outputs/calib_hdr.hdr')

eq_image = panoramic_transform(hdr_mirrorball_image)

write_hdr_image(eq_image, 'images/outputs/equirectangular.hdr')

plt.figure(figsize=(15,15))
display_hdr_image(eq_image, True)
                                              
O = read_image('images/outputs/proj4_objects.png')
E = read_image('images/outputs/proj4_empty.png')
M = read_image('images/outputs/proj4_mask.png')
M = M > 0.5
I = background_image
I = cv2.resize(I, (M.shape[1], M.shape[0]))
            
result = (M*O + (1-M)*I + (1-M)*(O-E)*0.1)

plt.figure(figsize=(20,20))
plt.imshow(result)
plt.show()

write_image(result, 'images/outputs/final_composite.png')