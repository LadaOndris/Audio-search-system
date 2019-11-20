# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 16:14:32 2019

@author: Ladislav Ondris

ISS project
"""
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import IPython
from scipy.signal import spectrogram, lfilter, freqz, tf2zpk
import scipy
import math
import sys
np.set_printoptions(threshold=sys.maxsize)

"""
plt.figure(figsize=(6,3))
plt.plot(t, data)
"""
Fs = 16000; 
N = 512; 
wlen = 25e-3 * Fs; 
wshift = 10e-3 * Fs; 
woverlap = wlen - wshift;

def show_spectogram(f, t, sgr):
    # prevod na PSD
    # (ve spektrogramu se obcas objevuji nuly, ktere se nelibi logaritmu, proto +1e-20)
    sgr_log = 10 * np.log10(sgr+1e-20) 
    
    plt.figure(figsize=(10,2))
    plt.pcolormesh(t, f, sgr_log)
    plt.gca().set_xlabel('Time [s]')
    plt.gca().set_ylabel('Frequency [Hz]')
    cbar = plt.colorbar()
    cbar.set_label('Spectral power density [dB]', rotation=270, labelpad=15)
    
    plt.tight_layout()
    #plt.savefig('../sx186_spectogram.png')

def compute_features(f, t, sgr):
    Nc = 16
    B = f.size // Nc  #256 / Nc
    print(B)
    A = np.zeros((B, f.size))
    print(A.shape)
    for i in range(B):
        for j in range(Nc):
            A[i][i * Nc + j] = 1
    print(A.shape)
    features = np.matmul(A, sgr)
    return B, features

"""
Q are query parameters
F are feature parameters
"""
def compute_correlation(Q, F, pp):
    Q_transposed = np.transpose(Q)
    F_transposed = np.transpose(F)
    
    correlation = 0
    for i in range(Q[0].shape[0]): # Q shape is (43, 16)
        corr, p_value = scipy.stats.pearsonr(Q_transposed[i], F_transposed[i + pp])
        if not math.isnan(corr):
            correlation += corr
    return correlation
    
def get_features(data, fs): 
    t = np.arange(data.size) / fs
    f, t, sgr = spectrogram(data, fs, nperseg=N, noverlap=woverlap)
    
    B, features = compute_features(f, t, sgr)
    show_spectogram(range(B), t, features)
    return B, t, features

data_s, fs_s = sf.read('../sentences/si1446.wav')
data_q, fs_q = sf.read('../queries/q1.wav')

F_frames_count, F_t, F = get_features(data_s, fs_s)
Q_frames_count, Q_t, Q = get_features(data_q, fs_q)

F_length = F[0].shape[0]
Q_length = Q[0].shape[0]
total_steps = F_length - Q_length
print(F_length, Q_length)

correlations = np.zeros(F_length)
for i in range(total_steps):
    corr = compute_correlation(Q, F, i)
    correlations[i] = corr
    print(corr)
    
    
fig, axs = plt.subplots(3, 1)
axs[0].plot(np.arange(data_s.size) / Fs, data_s)

sgr_log = 10 * np.log10(F + 1e-20) 
axs[1].pcolormesh(F_t, range(F_frames_count), sgr_log)

axs[2].plot(F_t, correlations)

fig.tight_layout()
plt.show()

























