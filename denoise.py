import numpy as np
from scipy.stats import iqr 

#~O(n log n) runtime

# denoising performed using FFT and FFT^-1 channels
# Input: 1-d array
# Output: 1-d denoised array by way of FFT + InvFFT
def denoise(sig):
    n = len(sig)
    F = np.fft.fft(sig, n) #compute FFT 
    pwrSpec = (F*np.conj(F))/n
    #IQR to determine outliers 
    iqrange = iqr(sig)
    Q3 = np.quantile(sig, 0.75)
    highlier = Q3 + 0.5 * iqrange
    #remove noise
    #pwrSpec2 = pwrSpec*(pwrSpec > highlier)
    F2 = F*(pwrSpec > highlier)
    return np.real(np.fft.ifft(F2)) #cleaned signal output