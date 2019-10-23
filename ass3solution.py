import numpy as np
from scipy.io.wavfile import read
from scipy.signal import find_peaks
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
    K = len(xb[1])
    X = np.abs(np.fft.rfft(xb*hann(K), K, axis=1))
    val = (1.0 * fs) / K
    freq = np.arange(0, len(X[0]), dtype=int) * val
    return X, freq

# A2
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

# B1
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

# B2
def track_pitch_hps(x, blockSize, hopSize, fs):
    xb, t=block_audio(x, blockSize, hopSize, fs)
    order=4
    X, freq=compute_spectrogram(xb, fs)
    f0=get_f0_from_Hps(X, fs, order)
    timeInSec=t
    return f0, timeInSec

# C1
def extract_rms(xb):
    rms = np.zeros(xb.shape[0])
    for i in range(xb.shape[0]):
        rms[i] = np.sqrt(np.mean((xb[i, :] ** 2)))
    e = 0.00001
    rms[rms < e] = e
    rms = 20 * np.log10(rms)

# C2
def create_voicing_mask(rmsDb, thresholdDb):
    mask = rmsDb
    for i in range(rmsDb.shape[0]):
        if rmsDb[i] < thresholdDb:
            mask[i] = 0
        elif rmsDb[i] >= thresholdDb:
            mask[i] = 1
    return mask

# C3
def apply_voicing_mask(f0, mask):

    f0Adj = f0.squeeze() * mask.squeeze()
    return f0Adj

# D1
def eval_voiced_fp(estimation, annotation):
    pfp = np.count_nonzero(estimation) / np.count_nonzero(annotation==0)
    return pfp

# D2
def eval_voiced_fn(estimation, annotation):
    num = np.take(estimation, np.nonzero(annotation))
    pfn = np.count_nonzero(num==0) / np.count_nonzero(annotation)
    return pfn

# ACF
def get_f0_from_acf(inputVector, fs, bIsNormalized=True):
    r = np.correlate(inputVector, inputVector, 'full')

    if bIsNormalized:
        r = r/(np.sum(np.square(r)))

    r =  r[len(r) // 2 :]
    peaks = find_peaks(r)[0]
    # plt.plot(r)
    # plt.plot(peaks, r[peaks], 'rs')
    # plt.show()
    if len(peaks) >= 2:
        p = sorted(r[peaks])[::-1]
        sorted_arg = np.argsort(r[peaks])[::-1]
        f0 = fs / abs(peaks[sorted_arg][1] - peaks[sorted_arg][0])
        return f0
    return 0

def track_pitch_acf(x, blockSize, hopSize, fs):
    blocked_x, timeInSec = block_audio(x, blockSize, hopSize, fs)
    frequencies = []
    for b in blocked_x:
        f0 = get_f0_from_acf(b, fs)
        frequencies.append(f0)
    return np.array(frequencies), timeInSec


# D3
def eval_pitchtrack_v2(estimation, annotation):


    pfp = eval_voiced_fp(estimation, annotation)
    pfn = eval_voiced_fn(estimation, annotation)

    # Remove values for zeros in groundtruthInHz
    estimation = np.take(estimation, np.nonzero(annotation)).squeeze()
    annotation = np.take(annotation, np.nonzero(annotation)).squeeze()

    # Remove values for zeros in estimateInHz
    annotation = np.take(annotation, np.nonzero(estimation)).squeeze()
    estimation = np.around(np.take(estimation, np.nonzero(estimation)).squeeze(), 3)

    # Convert to Cents
    estimateInCents = 1200 * np.log2(estimation / 440)
    annotationInCents = 1200 * np.log2(annotation / 440)

    errCentRms = np.sqrt(np.mean(np.power(estimateInCents - annotationInCents, 2)))

    return errCentRms, pfp, pfn


# E1
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

    print(t_fftmax)

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

    return

def eval(path_data):

    f0_means = []
    textFiles = sorted(glob.glob(os.path.join(path_data, '*.txt')))
    audioFiles = sorted(glob.glob(os.path.join(path_data, '*.wav')))
    # for Afile in audioFiles:
    #     fs, audio = wavread(file)
    #     f0, t = track_pitch_fftmax(audio, 1024, 512, fs)
    #     = eval_pitchtrack_v2(audioFiles[0], 1024, 512, 44100)
    #     plt.plot(t, f0)
    #     f0_avg = np.mean(f0)
    #     f0_means.append(f0_avg)
    for i in range(len(textFiles)):
        print('Processing file:', i + 1)

        # Calculate estimated pitch
        fs, audio = wavread(audioFiles[i])
        # est_freqHz, est_timestamps = track_pitch_fftmax(audio, 1024, 512, fs)
        _, est_timestamps = block_audio(audio, 1024, 512, fs)
        # Get annotated data
        with open(textFiles[i], 'r') as f:
            ann_timestamps = []
            ann_freqHz = []
            for eachLine in f:
                fields = eachLine[:-1].split('     ')
                ann_timestamps.append(float(fields[0]))
                ann_freqHz.append(float(fields[2]))

            ann_timestamps = np.array(ann_timestamps)
            ann_freqHz = np.array(ann_freqHz)

        matchPrecision = 0.5  # Check for timestamps in range of 1 sec
        matchingAnnotatedTimestamps = []
        matchingAnnotatedFreqHz = []

        for i in range(len(est_timestamps)):
            est_timestamps[i] = np.around(est_timestamps[i], 3)

            closebyTimestampsIdx = np.argwhere((ann_timestamps > (est_timestamps - matchPrecision)) & (
                        ann_timestamps < (est_timestamps + matchPrecision))).squeeze()

            closestTimestampIdx = sorted(closebyTimestampsIdx)[0] + np.abs(
                np.take(ann_timestamps, closebyTimestampsIdx) - est_timestamps).argmin()

            matchingAnnotatedTimestamps.append(ann_timestamps[closestTimestampIdx])
            matchingAnnotatedFreqHz.append(ann_freqHz[closestTimestampIdx])

        matchingAnnotatedTimestamps = np.array(matchingAnnotatedTimestamps)
        matchingAnnotatedFreqHz = np.array(matchingAnnotatedFreqHz)
        est_freqHz_fft, est_timestamps_fft = track_pitch_fftmax(audio, 1024, 512, fs)
        errCentRms_fft, pfp_fft, pfn_fft = eval_pitchtrack_v2(np.transpose(est_freqHz_fft), ann_freqHz)
        est_freqHz_hps, est_timestamps_hps = track_pitch_hps(audio, 1024, 512, fs)
        errCentRms_hps, pfp_hps, pfn_hps = eval_pitchtrack_v2(np.transpose(est_freqHz_hps), ann_freqHz)
    return



if __name__ == '__main__':
    # f = executeassign3('trainData')
    # print(f)
    # executeassign3()
    # eval('trainData')

    fs, x = wavread('./traindata/sine-440.wav')
    xb,_ = block_audio(x, 1024, 512, fs)



