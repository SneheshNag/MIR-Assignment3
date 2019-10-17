import numpy as np
from scipy.io.wavfile import read
import matplotlib.pyplot as plt


def wavread(path):
    sr, x = read(path)
    if x.dtype == 'float32':
        return sr, x
    elif x.dtype == 'uint8':
        return sr, (x / 128.) - 1
    else:
        bits = x.dtype.itemsize * 8
        return sr, x / (2 ** (bits - 1))

def block_audio(x, blockSize, hopSize, fs):
    numBlocks = int(np.ceil(x.size / hopSize))
    xb = np.zeros([numBlocks, blockSize])
    t = (np.arange(numBlocks) * hopSize) / fs
    x = np.concatenate((x, np.zeros(blockSize)), axis=0)
    for n in range(numBlocks):
        i_start = int(n * hopSize)
        i_stop = int(np.min([x.size - 1, i_start + blockSize - 1]))
        xb[n][np.arange(0, blockSize)] = x[np.arange(i_start, i_stop + 1)]
    return xb, t

def hann(window_length):
    return 0.5 - (0.5 * np.cos(2 * np.pi / window_length * np.arange(window_length))).reshape(1, -1)

# A1
def compute_spectrogram(xb, fs):
    K = len(xb[1])
    X = np.abs(np.fft.rfft(xb*hann(K), K, axis=1))
    val = (1.0 * fs) / K
    freq = np.arange(0, len(X[0]), dtype=int) * val
    return X, freq

def track_pitch_fftmax(x, blockSize, hopSize, fs):
    xb, t=block_audio(x, blockSize, hopSize, fs)
    X, freq=compute_spectrogram(xb, fs)
    timeInSec=t
    f0=np.zeros((1,len(X)))
    for i in range(len(X)):
        for j in range(len(X[0])-1):
            if max(X[i,:])==X[i,j]:
                f0[0,i]=freq[j]
    return f0, timeInSec

def get_f0_from_Hps(X, fs, order):
    freqRange=int(len(X[0])/order)
    HPS=X[:,0:freqRange]
    f0=np.zeros((1,len(X)))
    freqSpread=np.linspace(0,fs/2, len(X[0]))
    for h in range(len(X)):
        for j in range(1,order):
            for i in range(freqRange):
                HPS[h,i]=HPS[h,i]*(X[h,i*(j+1)]**2)  
            if max(HPS[h,:])>10**100:
                HPS[h,:]=HPS[h,:]/max(HPS[h,:])  
        for j in range(freqRange):
            if max(HPS[h,:])==HPS[h,j]:
                index=j
        f0[0,h]=freqSpread[index]      
    return f0

def track_pitch_hps(x, blockSize, hopSize, fs):
    xb, t=block_audio(x, blockSize, hopSize, fs)
    order=4
    X, freq=compute_spectrogram(xb, fs)
    f0=get_f0_from_Hps(X, fs, order)
    timeInSec=t
    return f0, timeInSec
