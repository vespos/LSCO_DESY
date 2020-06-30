import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapz
from scipy.optimize import curve_fit
from scipy import ndimage
import skbeam.core.correlation as corr
from pathlib import Path

from xpcs_ana import *


def get_file_name(run, data_dir):
    folder = data_dir / Path('lsco125_s1_28K_{:05d}/e4m/'.format(run))
    fname_eiger = folder / Path('lsco125_s1_28K_{:05d}_data_000001.h5'.format(run))
    fname_master = folder / Path('lsco125_s1_28K_{:05d}_master.h5'.format(run))
    fname_batchinfo = folder / Path('lsco125_s1_28K_{:05d}.batchinfo'.format(run))
    
    if not fname_batchinfo.exists():
        fname_batchinfo = None
        
    if not fname_eiger.exists() or not fname_master.exists():
        if not folder.exists:
            print('Folder does not exsits')
        else:
            print('data file does not exists')
    return fname_eiger, fname_master, fname_batchinfo


def fit_LS(func,x,y,yerr,initial_guess):  # fit the curve
    popt,pcov = curve_fit(func, x, y, p0=initial_guess, sigma=yerr, maxfev=1000000)
    popt,pcov = curve_fit(func, x, y, p0=popt, sigma=yerr, maxfev=1000000)
    perr = np.sqrt(np.diag(pcov))
    fitted_parameters = np.zeros([len(initial_guess), 2])
    fitted_parameters[:, 0] = popt
    fitted_parameters[:, 1]= perr
    return fitted_parameters


def centering_ls(th_imgs, drow1, drow2, dcol1, dcol2):
    # centering the theta scan and get the pixel coordinates of the Bragg point.
    image_mean = th_imgs.mean(axis=0) # take the mean
    image_mean[image_mean>1e7] = 0 # kill the gaps
    image_mean[image_mean<=0] = 0 # dont like zeros
    idx = np.where(image_mean==image_mean.max()) # find the maximum pixel
    qoi = [idx[0][0]-drow1,idx[0][0]+drow2,idx[1][0]-dcol1,idx[1][0]+dcol2] # define the roi for further processing
    
    my_images = th_imgs[:,qoi[0]:qoi[1],qoi[2]:qoi[3]] # only use the data within the roi
    my_images[my_images>=1e6] = 0.0
    size = my_images.shape
    intensity = my_images.sum(axis=1).sum(axis=1)
    x = np.arange(intensity.shape[0])

    fun = lorentzian
    guess = [5e6, 51.0, 1.0]
    params = fit_LS(fun, x, intensity, np.sqrt(intensity), guess)
#     print(params)
    ioi = int(round(params[1,0])) # go to the center image, this is also the samth for time scan
    print('Image number of interest: {}'.format(ioi))
    fig, ax = plt.subplots(ncols=2)
    ax[1].plot(x, intensity, 'o')
    ax[1].plot(x, fun(x, *params[:,0]))
    
    my_image = th_imgs[ioi, qoi[0]:qoi[1], qoi[2]:qoi[3]]*1.0
    my_image[my_image>1e7] = 0
    my_image[my_image<=0] = 0
    com = np.round(ndimage.center_of_mass(my_image)).astype(int) # find the center of mess
    print('Center of Mass of the image : {0:d}, {1:d}'.format(com[0],com[1]))
    ax[0].imshow(my_image[com[0]-100:com[0]+100, com[0]-100:com[0]+100])
    plt.tight_layout()
    plt.show()
    
#     x = np.arange(my_image.shape[0])*1.0
#     y = my_image[:,com[1]-2:com[1]+3].mean(axis=1)
#     e = np.sqrt(my_image[:,com[1]-2:com[1]+3].mean(axis=1))/2.3
#     e[e<1]=1
#     guess = [1e6,com[0],5.]
#     params = fit_LS(gaus,x,y,e,guess)
#     com[0] = int(round(params[1,0]))
#     x = np.arange(my_image.shape[0])
#     y = my_image[com[0]-2:com[0]+3,:].mean(axis=0)
#     e = np.sqrt(my_image[com[0]-2:com[0]+3,:].mean(axis=0))/2.3
#     e[e<1]=1
#     guess = [1e6,com[1],5.]
#     params = fit_LS(gaus,x,y,e,guess)
#     com[1] = int(round(params[1,0]))
#     print('Fine center of Mass of the image : {0:d}, {1:d}'.format(com[0],com[1]))
    return qoi,com


def roi_lsco(eiger_img, com, r0, dr, phi0, dphi, tilt):
    rad, phi = polarCoord(eiger_img.shape, com)
    ra = r0-dr
    rb = r0+dr
    phi0 = phi0 + tilt
    phia = phi0-dphi
    phib = phi0+dphi
    if phib>180:
        phib = phib-360
    roi = polarROI(rad, phi, ra, rb, phia, phib)
    return roi


def info_reader_ls(fbatchinfo):
    # read the information file of a run.
    my_file_name = fbatchinfo
    my_file=open(my_file_name, 'r').readlines()
    number_of_lines=len(my_file)
    my_data = []
    for i in range(number_of_lines):
        lines=my_file[i].split()
        if lines[0]=='start_time:':
            break
    tmin = int('2020'+'03'+str(lines[3][:2])+str(lines[4][:2])+str(lines[4][3:5])+str(lines[4][6:8]))
    return tmin