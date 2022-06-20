from python_speech_features import mfcc
import glob
import os
import scipy.io.wavfile as wav
import numpy
from python_speech_features import sigproc

def calculate_nfft(samplerate, winlen):
    window_length_samples = winlen * samplerate
    nfft = 1
    while nfft < window_length_samples:
        nfft *= 2
    return nfft


def fbank(signal,samplerate=16000,winlen=0.025,winstep=0.01,
          nfilt=26,nfft=512,lowfreq=0,highfreq=None,preemph=0.97,
          winfunc=lambda x:numpy.ones((x,))):
    highfreq= highfreq or samplerate/2
    signal = sigproc.preemphasis(signal,preemph)
    frames = sigproc.framesig(signal, winlen*samplerate, winstep*samplerate, winfunc)
    pspec = sigproc.powspec(frames,nfft)
    energy = numpy.sum(pspec,1) # this stores the total energy in each frame
    energy = numpy.where(energy == 0,numpy.finfo(float).eps,energy) # if energy is zero, we get problems with log

    fb = get_filterbanks(nfilt,nfft,samplerate,lowfreq,highfreq)
    feat = numpy.dot(pspec,fb.T) # compute the filterbank energies
    feat = numpy.where(feat == 0,numpy.finfo(float).eps,feat) # if feat is zero, we get problems with log

    return feat,energy


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

def hz2mel(hz):
    return 1127 * numpy.log(1+hz/700.0)

def mel2hz(mel):
    return 700 * (numpy.exp(mel/1127.0)-1)
def lifter(cepstra, L=22):
    """Apply a cepstral lifter the the matrix of cepstra. This has the effect of increasing the
    magnitude of the high frequency DCT coeffs.
    :param cepstra: the matrix of mel-cepstra, will be numframes * numcep in size.
    :param L: the liftering coefficient to use. Default is 22. L <= 0 disables lifter.
    """
    if L > 0:
        nframes,ncoeff = numpy.shape(cepstra)
        n = numpy.arange(ncoeff)
        lift = 1 + (L/2.)*numpy.sin(numpy.pi*n/L)
        return lift*cepstra
    else:
        # values of L <= 0, do nothing
        return cepstra


def mfcc_extraction(dir_path,audio_format='wav'):
    files=glob.glob(os.path.join(dir_path,'*.'+audio_format))
    if audio_format=='wav':
        cntr=0
        for aud_file in files:
            (rate,sig)=wav.read(aud_file)
            mfcc_feature=mfcc(sig,rate)
            return mfcc_feature
            cntr+=1
        print('{} {} files extract mfcc feature and saved in {}'.format(cntr,audio_format,dir_path))


i=0
for i in range(1, 401):
    myfile='clip_'+str(i)
    mfcc_extracted=mfcc_extraction('D:\\0001-0400\\0001-0400\\'+myfile, 'wav') #  영상 400개 예시로 mfcc 뽑은 것임.
    print('mfcc추출:', mfcc_extracted)
    print('mfcc shape:', mfcc_extracted.shape) #열은 13이고 행은 다 다름.
    i+=1
    #  (8934,13)도 있고, (10490,13)도 있고, (8112,13)도 있음.
