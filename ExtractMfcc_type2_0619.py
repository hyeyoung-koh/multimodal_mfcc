from python_speech_features import mfcc
import glob
import os
import scipy.io.wavfile as wav
import numpy
from python_speech_features import sigproc
from scipy.fftpack import dct


def calculate_nfft(samplerate, winlen):
    window_length_samples = winlen * samplerate
    nfft = 1
    while nfft < window_length_samples:
        nfft *= 2
    return nfft


def mfcc(signal, samplerate=16000, winlen=0.025, winstep=0.01, numcep=13,
         nfilt=26, nfft=None, lowfreq=0, highfreq=None, preemph=0.97, ceplifter=22, appendEnergy=True,
         winfunc=lambda x: numpy.ones((x,))):
    nfft = nfft or calculate_nfft(samplerate, winlen)
    feat, energy = fbank(signal, samplerate, winlen, winstep, nfilt, nfft, lowfreq, highfreq, preemph, winfunc)
    feat = numpy.log(feat)
    feat = dct(feat, type=2, axis=1, norm='ortho')[:, :numcep]
    feat = lifter(feat, ceplifter)
    if appendEnergy: feat[:, 0] = numpy.log(energy)  #  replace first cepstral coefficient with log of frame energy
    return feat


def fbank(signal, samplerate=16000, winlen=0.025, winstep=0.01,
          nfilt=26, nfft=512, lowfreq=0, highfreq=None, preemph=0.97,
          winfunc=lambda x:numpy.ones((x,))):
    highfreq= highfreq or samplerate/2
    signal = sigproc.preemphasis(signal, preemph)
    frames = sigproc.framesig(signal, winlen*samplerate, winstep*samplerate, winfunc)
    pspec = sigproc.powspec(frames, nfft)
    energy = numpy.sum(pspec, 1) # this stores the total energy in each frame
    energy = numpy.where(energy == 0, numpy.finfo(float).eps, energy) # if energy is zero, we get problems with log

    fb = get_filterbanks(nfilt,nfft, samplerate,lowfreq,highfreq)
    feat = numpy.dot(pspec,fb.T) # compute the filterbank energies
    feat = numpy.where(feat == 0,numpy.finfo(float).eps,feat) # if feat is zero, we get problems with log

    return feat,energy

def logfbank(signal,samplerate=16000,winlen=0.025,winstep=0.01,
          nfilt=40,nfft=512,lowfreq=64,highfreq=None,dither=1.0,remove_dc_offset=True,preemph=0.97,wintype='hamming'):
    feat,energy = fbank(signal,samplerate,winlen,winstep,nfilt,nfft,lowfreq,highfreq,dither, remove_dc_offset,preemph,wintype)
    return numpy.log(feat)

def hz2mel(hz):
    return 1127 * numpy.log(1+hz/700.0)


def mel2hz(mel):
    return 700 * (numpy.exp(mel/1127.0)-1)

def get_filterbanks(nfilt=26,nfft=512,samplerate=16000,lowfreq=0,highfreq=None):
    highfreq= highfreq or samplerate/2
    assert highfreq <= samplerate/2, "highfreq is greater than samplerate/2"

    # compute points evenly spaced in mels
    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)

    # check kaldi/src/feat/Mel-computations.h
    fbank = numpy.zeros([nfilt,nfft//2+1])
    mel_freq_delta = (highmel-lowmel)/(nfilt+1)
    for j in range(0,nfilt):
        leftmel = lowmel+j*mel_freq_delta
        centermel = lowmel+(j+1)*mel_freq_delta
        rightmel = lowmel+(j+2)*mel_freq_delta
        for i in range(0,nfft//2):
            mel=hz2mel(i*samplerate/nfft)
            if mel>leftmel and mel<rightmel:
                if mel<centermel:
                    fbank[j,i]=(mel-leftmel)/(centermel-leftmel)
                else:
                    fbank[j,i]=(rightmel-mel)/(rightmel-centermel)
    return fbank

def lifter(cepstra, L=22):
    if L > 0:
        nframes,ncoeff = numpy.shape(cepstra)
        n = numpy.arange(ncoeff)
        lift = 1 + (L/2.)*numpy.sin(numpy.pi*n/L)
        return lift*cepstra
    else:
        # values of L <= 0, do nothing
        return cepstra

def delta(feat, N):
    if N < 1:
        raise ValueError('N must be an integer >= 1')
    NUMFRAMES = len(feat)
    denominator = 2 * sum([i**2 for i in range(1, N+1)])
    delta_feat = numpy.empty_like(feat)
    padded = numpy.pad(feat, ((N, N), (0, 0)), mode='edge')   # padded version of feat
    for t in range(NUMFRAMES):
        delta_feat[t] = numpy.dot(numpy.arange(-N, N+1), padded[t : t+2*N+1]) / denominator   # [t : t+2*N+1] == [(N+t)-N : (N+t)+N+1]
    return delta_feat

def mfcc_extraction(dir_path,audio_format='wav'):
    files=glob.glob(os.path.join(dir_path,'*.'+audio_format))
    if audio_format=='wav':
        cntr=0
        for aud_file in files:
            (rate,sig)=wav.read(aud_file)
            global mfcc_feature
            mfcc_feature=mfcc(sig,rate)
            cntr+=1
        print(mfcc_feature)


#예시로 클립 100개에 대한 mfcc추출
i=0
for i in range(1, 101):
    myfile = 'clip_'+str(i)
    mfcc_extraction('D:\\0001-0400\\0001-0400\\'+myfile, 'wav')
    numpy.savetxt('D:\\0001-0400\\0001-0400\\'+myfile+'.csv', mfcc_feature, delimiter=",")
    i+=1

