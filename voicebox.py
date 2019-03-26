"""
some tool func about speech processing 

"""


import scipy 
import scipy.signal as signal 
import numpy as np
import wave
import scipy.io as sio

def audioread(path, sample_rate=16000):
    """
        read wave data like matlab's audioread
        selected_channels: for multichannel wave, return selected_channels' data 
    """
    with wave.open(path, 'rb') as fid:
        params = fid.getparams()
        nchannels, samplewidth, framerate , nframes = params[:4]
        assert nchannels == 1
        strdata = fid.readframes(nframes)
        wavedata = np.fromstring(strdata, dtype=np.int16)
        wavedata = wavedata*1.0/(32767.0*samplewidth/2)
        wavedata = np.reshape(wavedata, [nframes]) 
    return wavedata

def audiowrite(path, data, nchannels=1, samplewidth=2, framerate=16000):

    nframes = len(data)
    with wave.open(path, 'wb') as fid:
        data *= 32767.0
        fid.setparams((nchannels, samplewidth, framerate, nframes,"NONE", "not compressed"))
        fid.writeframes(np.array(data, dtype=np.int16).tostring())

def enframe(data, window, win_len, inc):
    data_len = data.shape[0] 
    if data_len <= win_len :
        nf = 1
    else:
        nf = int(np.ceil((1.0*data_len-win_len+inc)/inc))
    pad_length = int((nf-1)*inc+win_len)
    zeros = np.zeros((pad_length - data_len, ))
    pad_signal = np.concatenate((data, zeros))
    indices = np.tile(np.arange(0,win_len), (nf,1))+ np.tile(np.arange(0,nf*inc, inc), (win_len,1)).T 
    indices = np.array(indices, dtype=np.int32)
    frames = pad_signal[indices]
    windows = np.reshape(np.tile(window, nf),[nf,win_len])
    return frames*windows

def fft(data, fft_len):
    return np.abs(np.fft.rfft(data, n=fft_len)) 

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

def activelev(data,fs):
    """
        Normalize data to 0db
        now is a little func,
        furthermore it will be
        transplantated from matlab version
    """
    max_amp = np.max(np.abs(data))
    return data/max_amp

def resample(src, fs, tfs):
    if fs == tfs:
        return src
    if fs > tfs:
        down_sample(src, fs, tfs)
    else:
        up_sample(src, fs, tfs)

def up_sample(src, fs, tfs):
    """
        up sample
    """
    pass

def down_sample(src, fs, tfs):
    """
        down sample
    """
    pass

def AddNoise(clist, nlist, out_clean_wav, out_noisy_wav, out_specs, log):
    with open(clist) as cfid:
        with open(nlist) as nfid:
            with open(log, 'w') as lfid:
                for line in cfid:



if __name__ == '__main__':
    wave_data = audioread('./000001.wav') 
    win = np.hamming(400)/1.2607934
    en_data = enframe(wave_data,win,400, 100)
    fft_data=fft(en_data,512)

    sio.savemat('test.mat', {'py_en':en_data, 'py_fft':fft_data, 'py_wave':wave_data})
