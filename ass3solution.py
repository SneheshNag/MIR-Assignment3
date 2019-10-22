import numpy as np
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
import os
import glob


def wavread(path):
    fs, x = read(path)
    if x.dtype == 'float32':
        return fs, x
    elif x.dtype == 'uint8':
        return fs, (x / 128.) - 1
    else:
        bits = x.dtype.itemsize * 8
        return fs, x / (2 ** (bits - 1))

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
    K = len(xb[0])
    X = np.abs(np.fft.rfft(xb*hann(K), K, axis=1))
    val = (1.0 * fs) / K
    freq = np.arange(0, len(X[0]), dtype=int) * val
    return X, freq

def track_pitch_fftmax(x, blockSize, hopSize, fs):
    xb, t = block_audio(x, blockSize, hopSize, fs)
    X, freq=compute_spectrogram(xb, fs)
    timeInSec=t
    f0=np.zeros((1,len(X)))
    for i in range(len(X)):
        for j in range(len(X[0])-1):
            if max(X[i,:])==X[i,j]:
                f0[0,i]=freq[j]
    return f0, timeInSec

def get_f0_from_Hps(X, fs, order):
    freqRange=int((len(X[0])-1)/order)
    # print len(X)
    # print X.shape
    f0=np.zeros((1,len(X)))
    HPS=np.zeros((len(X),freqRange))
    freqSpread=np.linspace(0,fs/2, len(X[0]))
    for h in range(len(X)):
        for i in range(freqRange):
            multiplier=1
            for j in range(1,order+1):
                multiplier=multiplier*(X[h,i*j])    #There should be a power of 2 here but it seemed to work better without it
            HPS[h,i]=multiplier
            if max(HPS[h,:])>10**10:    #Prevent the HPS getting too big 
                HPS[h,:]=HPS[h,:]/max(HPS[h,:]) 
        for j in range(freqRange):
            if max(HPS[h,:])==HPS[h,j]:
                index=j
        f0[0,h]=freqSpread[index] 
    return f0

def track_pitch_hps(x, blockSize, hopSize, fs):
    xb, t=block_audio(x, blockSize, hopSize, fs)
    order=7
    X, freq=compute_spectrogram(xb, fs)
    f0=get_f0_from_Hps(X, fs, order)
    timeInSec=t
    return f0, timeInSec

def extract_rms(xb):
    rms = np.zeros(xb.shape[0])
    for i in range(xb.shape[0]):
        rms[i] = np.sqrt(np.mean(xb[i]**2))  #needs to be corrected
    e = 0.00001
    rms[rms < e] = e
    rms = 20 * np.log10(rms)
    return rms

def create_voicing_mask(rmsDb, thresholdDb):
    mask = rmsDb
    for i in range(rmsDb.shape[0]):
        if rmsDb[i] < thresholdDb:
            mask[i] = 0
        elif rmsDb[i] >= thresholdDb:
            mask[i] = 1
    return mask

def apply_voicing_mask(f0, mask):

    f0Adj = f0 * mask
    return f0Adj

def executeassign3():
    f1 = 441.
    f2 = 882.
    fs = 44100
    y1 = np.sin(2 * np.pi * f1 * np.linspace(0, 1, fs))
    y2 = np.sin(2 * np.pi * f2 * np.linspace(1, 2, fs))
    y = np.concatenate((y1, y2), axis=0)
    f0_fftmax, t_fftmax = track_pitch_fftmax(y, 2048, 512, fs)
    print(f0_fftmax.shape, t_fftmax.shape)
    f0_hps, t_hps = track_pitch_hps(y, 1024, 512, fs)
    f0_ref = np.concatenate((np.ones(int(np.floor(len(t_fftmax) / 2))) * 441, np.ones(int(np.ceil(len(t_fftmax) / 2))) * 882), axis=0)
    error_fftmax = f0_fftmax - f0_ref
    error_hps = f0_hps - f0_ref
    # plt.plot(x, y)
    # plt.xlabel('sample(n)')
    # plt.ylabel('voltage(V)')
    # plt.show()
    # print(f0_fftmax)
    plt.subplot(211)
    plt.title('Predicted Frequency')
    plt.plot(t_fftmax, np.transpose(f0_fftmax), label='Prediction')
    plt.plot(t_fftmax, f0_ref, label='Ground Truth')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Frequency (Hz)')
    plt.grid()

    plt.subplot(212)
    plt.title('Prediction Error')
    plt.plot(t_fftmax, np.transpose(error_fftmax))
    plt.xlabel('Time')
    plt.ylabel('Error (Hz)')
    plt.grid()
    plt.tight_layout()
    plt.show()

    plt.subplot(211)
    plt.title('Predicted Frequency')
    plt.plot(t_hps, np.transpose(f0_hps), label='Prediction')
    plt.plot(t_hps, f0_ref, label='Ground Truth')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Frequency (Hz)')
    plt.grid()

    plt.subplot(212)
    plt.title('Prediction Error')
    plt.plot(t_hps, np.transpose(error_hps))
    plt.xlabel('Time')
    plt.ylabel('Error (Hz)')
    plt.grid()
    plt.tight_layout()
    plt.show()


def eval(path_data):

    f0_means = []
    txtFiles = sorted(glob.glob(os.path.join(path_data, '*.txt')))
    audioFiles = sorted(glob.glob(os.path.join(path_data, '*.wav')))
    for file in audioFiles:
        fs, audio = wavread(file)
        f0, t = track_pitch_fftmax(audio, 1024, 512, fs)
        plt.plot(t, f0)
        # f0_avg = np.mean(f0)
        # f0_means.append(f0_avg)

    return

def eval_pitchtrack_v2():
    
    track_pitch_hps(x, blockSize, hopSize, fs)

if __name__ == '__main__':
    # f = executeassign3('trainData')
    # print(f)
    executeassign3()


