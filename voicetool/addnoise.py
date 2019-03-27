import sys
import os

sys.path.append(os.path.dirname(sys.path[0]) + '/voicetoll')
import scipy.io as sio
import scipy
import numpy as np

def add_noisem(clean, noise, start, scale, snr, fs=16000):
    clean_size = clean.shape[0]
    noise_selected = noise[start:start+clean_size]
    clean_n = activelev(clean, fs)
    noise_n = activelev(noise_selected, fs)
    clean_snr = snr
    noise_snr = -snr
    clean_weight = 10**(clean_snr/20)
    noise_weight = 10**(noise_snr/20)
    clean = clean_n * clean_weight
    noise = noise_n * noise_weight
    noisy = clean + noise
    max_amp = np.max(np.abs([noise,clean,noisy]))
    mix_scale = 1/max_amp*scale
    X = clean * mix_scale
    N = noise * mix_scale
    Y = noisy * mix_scale
    return Y, X, N

def AddNoise(clist, nlist, out_clean_wav, out_noisy_wav, out_specs, log):
    with open(clist) as cfid:
        with open(nlist) as nfid:
            with open(log, 'w') as lfid:
                for line in cfid:

