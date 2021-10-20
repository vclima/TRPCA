import numpy as np
from scipy.signal import convolve2d

def matlab_style_gauss2D(shape=(11,11),sigma=1.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def ssim_index(img1, img2, K=[0.01, 0.03], window=None, L=255):
    """ Calculate the structural similarity between two images

    Args:
        img1 ([type]): [description]
        img2 ([type]): [description]
        K ([type]): [description]
        window ([type]): [description]
        L ([type]): [description]
    """
    if window is not None:
        window=matlab_style_gauss2D()
    
    if img1.shape != img2.shape:
        print("Error: Images has different shapes")
        return None
    
    M, N = img1.shape 

    if (M < 11 or N < 11):
        print("Error: wrong shape")
        return None
    
    # Maybe it's necessary to convert rgb to YCbCr

    C1 = (K[0]*L)**2
    C2 = (K[1]*L)**2

    window = window/window.sum()    # Normalize filter

    mu1 = convolve2d(img1, np.rot90(window), mode='valid')
    mu2 = convolve2d(img2, np.rot90(window), mode='valid')

    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2

    sigma1_sq = convolve2d(img1*img1, np.rot90(window), mode='valid')
    sigma2_sq = convolve2d(img2*img2, np.rot90(window), mode='valid')
    sigma12 = convolve2d(img1*img2, np.rot90(window), mode='valid')

    if(C1>0 and C2>0):
        ssim_map =  ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    else:
        numerator1 = 2*mu1_mu2 + C1
        numerator2 = 2*sigma12 + C2
        denominator1 = mu1_sq + mu2_sq + C1
        denominator2 = sigma1_sq + sigma2_sq + C2
        ssim_map = np.ones(mu1.shape)
        index = (denominator1*denominator2 > 0)
        ssim_map[index] = (numerator1[index]*numerator2[index])/(denominator1[index]*denominator2[index])
        index = (denominator1 != 0) & (denominator2 == 0)
        ssim_map[index] = numerator1[index]/denominator1[index]

    mssim = np.mean(ssim_map)

    return mssim