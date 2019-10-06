import numpy as np
from scipy.io.wavfile import read

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
        i_start = n * hopSize
        i_stop = np.min([x.size - 1, i_start + blockSize - 1])
        xb[n][np.arange(0, blockSize)] = x[np.arange(i_start, i_stop + 1)]
    return xb, t

def hann(window_length):
    return 0.5 - (0.5 * np.cos(2 * np.pi / window_length * np.arange(window_length))).reshape(1, -1)

# A1
def compute_spectrogram(xb, fs):
    K = len(xb[1])
    X = np.abs(np.fft.rfft(xb*hann(K), K, axis=1))
    val = (1.0 * fs) / K
    freq = np.arange(0, len(X), dtype=int) * val
    return X, freq
