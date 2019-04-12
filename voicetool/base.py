
"""
some tool func about speech processing 

author: yxhu
"""


import scipy 
import scipy.signal as signal 
import numpy as np
import wave
import scipy.io as sio

def audioread(path, sample_rate=16000, selected_channels=[1]):
    """
        read wave data like matlab's audioread
        selected_channels: for multichannel wave, return selected_channels' data 
    """
    with wave.open(path, 'rb') as fid:
        selected_channels = [ x - 1 for x in selected_channels]

        params = fid.getparams()
        nchannels, samplewidth, framerate , nframes = params[:4]
        strdata = fid.readframes(nframes*nchannels)
        wavedata = np.fromstring(strdata, dtype=np.int16)
        wavedata = wavedata*1.0/(32767.0*samplewidth/2)
        wavedata = np.reshape(wavedata, [nframes,nchannels])
        
    return wavedata[:, selected_channels]

def audiowrite(path, data, nchannels=1, samplewidth=2, framerate=16000):
    
    data = np.reshape(data, [-1, nchannels])
    nframes = data.shape[0]
    with wave.open(path, 'wb') as fid:
        data *= 32767.0
        fid.setparams((nchannels, samplewidth, framerate, nframes,"NONE", "not compressed"))
        fid.writeframes(np.array(data, dtype=np.int16).tostring())

def enframe(data, window, win_len, inc):
    data_len = data.shape[0] 
    if data_len <= win_len :
        nf = 1
    else:
        nf = int((data_len-win_len+inc)/inc)
    # 2019-3-29:
    # remove the padding, the last points will be discard

    #pad_length = int((nf-1)*inc+win_len)
    #zeros = np.zeros((pad_length - data_len, ))
    #pad_signal = np.concatenate((data, zeros))

    indices = np.tile(np.arange(0,win_len), (nf,1))+ np.tile(np.arange(0,nf*inc, inc), (win_len,1)).T 
    indices = np.array(indices, dtype=np.int32)
    frames = data[indices]
    windows = np.reshape(np.tile(window, nf),[nf,win_len])
    return frames*windows

def fft(data, fft_len):
    return np.abs(np.fft.rfft(data, n=fft_len)) 


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



if __name__ == '__main__':
    wave_data = audioread('./000001.wav').reshape([-1])
    win = np.hamming(400)/1.2607934
    en_data = enframe(wave_data,win,400, 100)
    fft_data=fft(en_data,512)

    sio.savemat('test.mat', {'py_en':en_data, 'py_fft':fft_data, 'py_wave':wave_data})
