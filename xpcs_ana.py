import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import skbeam.core.correlation as corr


def gauss(x, area, center, width):
    return abs(area)*np.exp(-np.power(x - center, 2.) / (2 * np.power(width, 2.))) / (width * np.sqrt(2 * np.pi))


def lorentzian(x, a, x0, gam):
    return a * gam**2 / ( gam**2 + (x-x0)**2)


def polarCoord(img_size, center):
    x, y = np.arange(img_size[1]), np.arange(img_size[0])
    xx, yy = np.meshgrid(x,y)
    cx = center[0]
    cy = center[1]
    
    rad = np.sqrt((xx-cx)**2+(yy-cy)**2)
    phi = np.rad2deg(-np.arctan2(yy-cy,xx-cx))
    return rad, phi


def polarROI(rad, phi, ra, rb, phia, phib):
    """ Polar ROI
    
    Args:
        rad, phi: output of polarCoord
        ra, rb: radial range
        phia, phib: azimuthal range
    """
    if phib<0 and phib<phia:
        ROI = np.logical_and((ra<=rad)*(rad<=rb), (phia<=phi)*(phi<=180))
        ROI = ROI + np.logical_and((ra<=rad)*(rad<=rb), (phi<=phib))
    else:
        ROI = np.logical_and((ra<=rad)*(rad<=rb), (phia<=phi)*(phi<=phib))
    return ROI


def box_to_roi(imgs, mask):
    """ Reduce the imgs to the smallest box around the roi defined by mask.
    
    Args:
        imgs: stack of images or single image (ndim= 2 or 3)
        mask: mask for imgs. Shape must correspond to the shape of the images
    Return:
        Cropped image and mask
    """
    if imgs.ndim==2:
        imgs = imgs[np.newaxis,...]
    nframe = imgs.shape[0]
    posx,posy = np.where(mask)
    shape = np.array([posx.max()-posx.min()+1,posy.max()-posy.min()+1])
    
    maskn = np.zeros(shape)
    maskn[posx-posx.min(),posy-posy.min()] = 1.
    imgs_red = np.zeros([nframe, *maskn.shape])
    imgs_red[:, posx-posx.min(),posy-posy.min()] = imgs[:,posx,posy].copy()
    return np.squeeze(imgs_red), maskn


def box_to_roi_extend(imgs, mask, extend=10):
    """ Reduce the imgs to an extended box around the roi defined by mask. Does not
    apply the mask.
    
    Args:
        imgs: stack of images or single image (ndim= 2 or 3)
        mask: mask for imgs. Shape must correspond to the shape of the images
        extend: number of pixel for the extension in each direction
    Return:
        Cropped image and mask
    """
    if imgs.ndim==2:
        imgs = imgs[np.newaxis,...]
    posx,posy = np.where(mask)
    posx_min = posx.min()-extend
    posx_max = posx.max()+extend
    posy_min = posy.min()-extend
    posy_max = posy.max()+extend
    
    maskn = mask[posx_min:posx_max, posy_min:posy_max]
    imgs_red = imgs[:,posx_min:posx_max, posy_min:posy_max]
    return np.squeeze(imgs_red), maskn


def _spatial_correlation_fourier(fim1, fim2_star, fmask, fmask_star):
    A_num1 = np.fft.irfft2(fim1*fim2_star)
    A_num2 = np.fft.irfft2(fmask*fmask_star)
#     A_num2 *=A_num2>0.4 # remove Fourier components that are too small
    A = A_num1 * A_num2
    A_denom = np.fft.irfft2(fim1*fmask_star) * np.fft.irfft2(fim2_star*fmask)
    # make sure the normalization value isn't 0 otherwise the autocorr will 'explode'
    pos = np.where(np.abs(A_denom) != 0)
    A[pos] /= A_denom[pos]
    A = np.fft.fftshift(A)
    return A


def spatial_correlation_fourier(img, img2=None, mask=None):
    """ Compute spatial correlation between two images using Fourier transform.
    
    Args:
        img: first image
        img2: second image. If None, becomes the first image (autocorrelation)
        mask: mask spanning the region of interest
    
    Returns:
        A: 2d correlation matrix
    """
    if mask is None:
        mask = np.ones_like(img)
    if img2 is None:
        img2 = img
        
    # (i) restrain mask and imgs to a bounding box around the roi defined by the mask
    img, temp = box_to_roi(img, mask)
    img2, mask = box_to_roi(img2, mask)
    
    # (ii) compute the different terms
    fmask = np.fft.rfft2(mask)
    fmask_star = np.conjugate(fmask)
    fimg = np.fft.rfft2(img)
    fimg2_star = np.conjugate(np.fft.rfft2(img2))
    
    # (iii) compute correlation
    A = _spatial_correlation_fourier(fimg, fimg2_star, fmask, fmask_star)
    return A


def remove_central_corr(A):
    i,j = np.unravel_index(np.argmax(A), A.shape)
    A[i,j] = np.nan
    return A


def correct_illumination(imgs, roi, kernel_size=5):
    """ Correct the detector images for non-uniform illumination.
    This implementaion follows Part II in Duri et al. PHYS. REV. E 72, 051401 (2005).
    
    Args:
        imgs: stack of detector images
        roi: region of interest to consider. Important so that the normalization of the correction 
            is ~unity for the specific roi
        kernel_size: size of the kernel for box-average. Can be None, in which case no kernel is 
            applied. The kernel is used to smooth out remaining speckly structure in the intensity correction.
        
    Returns:
        imgs: corrected images, cropped to an extended box aroung the roi
        roi: new roi for the cropped image
        bp: correction factor
    """
    if kernel_size is None:
        extend = 10
    else:
        extend=2*kernel_size
    imgs, roi = box_to_roi_extend(imgs, roi, extend=extend)
    bp = np.mean(imgs, axis=0)
    if kernel_size is not None:
        kernel = np.ones([kernel_size, kernel_size])/kernel_size/kernel_size
        bp = ndimage.convolve(bp, kernel)
    bp = bp / bp[roi].mean()
    zero = bp==0
    bp[zero] = 1e-6
    imgs_corr = imgs/bp
    return imgs_corr, roi, bp