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
import glob
import os
from pathlib import Path
np.set_printoptions(threshold=sys.maxsize)

Fs = 16000
N = Fs * 25 // 1000; 
wlen = 25e-3 * Fs
wshift = 10e-3 * Fs
woverlap = wlen - wshift

def show_spectogram(f, t, sgr, title):
    # prevod na PSD
    # (ve spektrogramu se obcas objevuji nuly, ktere se nelibi logaritmu, proto +1e-20)
    #sgr_log = 10 * np.log10(sgr+1e-20) 
    
    plt.figure(figsize=(10,4))
    plt.pcolormesh(t, f, sgr)
    plt.gca().set_xlabel('Time [s]')
    plt.gca().set_ylabel('Frequency [Hz]')
    plt.gca().set_title(title)
    cbar = plt.colorbar()
    cbar.set_label('Spectral power density [dB]', rotation=270, labelpad=15)
    
    plt.tight_layout()
    plt.savefig('../%s_spectogram.png' % title)

def compute_features(f, t, sgr):
    Nc = 16
    B = f.size // Nc  #256 / Nc
    A = np.zeros((B, f.size))
    
    for i in range(B):
        for j in range(Nc):
            A[i][i * Nc + j] = 1
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
    return correlation / Q[0].shape[0]
    
def get_features(data, fs): 
    t = np.arange(data.size) / fs
    f, t, sgr = spectrogram(data, fs, nperseg=N, noverlap=woverlap, nfft=511)
    sgr = 10 * np.log10(sgr+1e-20) 
    
    B, features = compute_features(f, t, sgr)
    #show_spectogram(range(B), t, features)
    return B, t, features

def compute_correlations(F, Q):
    F_length = F[0].shape[0]
    Q_length = Q[0].shape[0]
    total_steps = F_length - Q_length
    
    correlations = np.zeros(F_length)
    for i in range(total_steps):
        corr = compute_correlation(Q, F, i)
        correlations[i] = corr
    return correlations

def draw_correlations(filename, Q1, Q2): 
    data_s, fs_s = sf.read(filename)
    
    F_frames_count, F_t, F = get_features(data_s, fs_s)
    
    correlations_q1 = compute_correlations(F, Q1)
    correlations_q2 = compute_correlations(F, Q2)
        
    fig, axs = plt.subplots(3, 1, figsize=(10,5))
    axs[0].set_title('"essentially" and "exercise" vs %s' % os.path.split(filename)[-1])
    axs[0].plot(np.arange(data_s.size) / Fs, data_s)
    axs[0].set_ylabel('signal')
    axs[0].margins(x=0)
    
    axs[1].pcolormesh(F_t, range(F_frames_count), F)
    axs[1].set_ylabel('features')
    
    axs[2].plot(F_t, correlations_q1)
    axs[2].plot(F_t, correlations_q2)
    axs[2].set_ylabel('scores')
    axs[2].set_xlabel('Time [s]')
    axs[2].legend(['essentially', 'exercise'])
    axs[2].margins(x=0)
    axs[2].set_ylim([0, 1])
    
    #plt.savefig('../docs/%s.png' % Path(filename).resolve().stem)
    
def task3_spectogram():
    filepath = '../sentences/sx186.wav'
    filename = Path(filepath).resolve().stem
    data_s, fs_s = sf.read('../sentences/sx186.wav')
    f, t, sgr = spectrogram(data_s, Fs, nperseg=N, noverlap=woverlap)
    show_spectogram(f, t, sgr, filename)
    
#task3_spectogram()

filenames = [f for f in glob.glob('../sentences/*.wav')]

data_q, fs_q = sf.read('../queries/q1.wav')
data_q2, fs_q2 = sf.read('../queries/q2.wav')

Q_frames_count, Q_t, Q = get_features(data_q, fs_q)
Q2_frames_count, Q2_t, Q2 = get_features(data_q2, fs_q2)

"""
data_s, fs_s = sf.read('../sentences/sa2.wav')
f, t, sgr = spectrogram(data_s, Fs, nperseg=N, noverlap=woverlap, nfft=511)
print(sgr.shape)
"""
for filename in filenames:
    draw_correlations(filename, Q, Q2)
    


























