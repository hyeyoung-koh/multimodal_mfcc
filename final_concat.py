from transformers import BertTokenizer, BertModel
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
#---------------
from python_speech_features import mfcc
#from glob import glob
import glob
import os
import scipy.io.wavfile as wav

#from __future__ import division
import numpy
from python_speech_features import sigproc
from scipy.fftpack import dct

def calculate_nfft(samplerate, winlen):
    window_length_samples = winlen * samplerate
    nfft = 1
    while nfft < window_length_samples:
        nfft *= 2
    return nfft
#-------------------------------------------------
#함수 정의

#mfcc함수
def mfcc(signal,samplerate=16000,winlen=0.025,winstep=0.01,numcep=13,
         nfilt=26,nfft=None,lowfreq=0,highfreq=None,preemph=0.97,ceplifter=22,appendEnergy=True,
         winfunc=lambda x:numpy.ones((x,))):
    nfft = nfft or calculate_nfft(samplerate, winlen)
    feat,energy = fbank(signal,samplerate,winlen,winstep,nfilt,nfft,lowfreq,highfreq,preemph,winfunc)
    feat = numpy.log(feat)
    feat = dct(feat, type=2, axis=1, norm='ortho')[:,:numcep]
    feat = lifter(feat,ceplifter)
    if appendEnergy: feat[:,0] = numpy.log(energy) # replace first cepstral coefficient with log of frame energy
    return feat

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

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
model = BertModel.from_pretrained('bert-base-multilingual-uncased')
#model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-uncased',num_labels=8,output_attentions=False,output_hidden_states=False)
#data=pd.read_csv('aihub_clip1-5200.csv',encoding='utf-8-sig')
#our_text=str(data['text_script'])
aihub_data=pd.read_csv('aihub_sample100.csv',encoding='utf-8-sig')
aihub_text=aihub_data['text_script']
#print(aihub_text)
#print(aihub_text[0])
list_last_hidden_states=[]
#print(aihub_text[64975])
for i in tqdm(range(0,100)): #인덱스가 0부터 64977이다.
    myfile='clip_'+str(i)
    mfcc_extracted=mfcc_extraction('D:\\0001-0400\\0001-0400\\'+myfile,'wav') #영상 400개 예시로 mfcc 뽑은 것임.
    #convert_file=convert_files_to_wav('D:\\0001-0400\\0001-0400\\'+myfile,'mp4')
    print('mfcc추출:',mfcc_extracted)
    #mfcc추출한 벡터=>mfcc_extracted
    #print('mfcc shape:',mfcc_extracted.shape) #열은 13이고 행은 다 다름.

    aihub_inputs=aihub_text[i]
    aihub_final_inputs=tokenizer(aihub_inputs,return_tensors='pt') #이게 text의 inputs이다.
    #outputs=model(**aihub_final_inputs)
    outputs = model(**aihub_final_inputs)
    last_hidden_states=outputs.last_hidden_state
    list_last_hidden_states.append(last_hidden_states) #우리의 cls 토큰은 list_last_hidden_states[0][0][0]이다.
    print('text 피처추출벡터:',list_last_hidden_states[i][0][0]) #text feature추출한 벡터=>list_last_hidden_states[0][0][0]이다.
    i+=1
#inputs=tokenizer(aihub_text,return_tensors='pt')
#inputs = tokenizer('나는 학교에 간다', return_tensors="pt")
#outputs = model(**inputs)
# print(list_last_hidden_states) #이러면 쭉 저장됨.
# print(list_last_hidden_states[0][0][0])
# print(list_last_hidden_states[1][0][0])
#class torch.nn.Dropout(p=0.5,inplace=False)
#aihub_data.iloc[:,9]
num_labels=8
hidden_size=768
#classifier_dropout=classifier_dropout()
dropout=torch.nn.Dropout()
classifier=torch.nn.Linear(hidden_size,8)
#labels=aihub_data.iloc[:,10]
#pooled_output=list_last_hidden_states[0][0][0]
#loss_fct=torch.nn.BCEWithLogitsLoss()
loss_fct=torch.nn.CrossEntropyLoss()
#loss_fct=torch.nn.MultiLabelSoftMarginLoss()
for i in tqdm(range(0,100)):
    #pooled_output=list_last_hidden_states[i][0][0]
    pooled_output=list_last_hidden_staets[i][0][0]+mfcc_extracted
    print('pooled_output:',pooled_output)
    print('pooled_output의 shape:',pooled_output.shape)
    pooled_output=dropout(pooled_output)

    #labels=torch.tensor(aihub_data.iloc[i,10]).unsqueeze(-1) #인덱스가 0부터 시작하므로 0은 첫번째 차원을 의미한다.
    labels = torch.tensor(aihub_data.iloc[i, 11])

    print('original labels:',labels) #labels출력하면 tensor([1])과 같은 형태로 출력됨
    #labels=torch.tensor(labels)
    #labels=torch.tensor([labels]).unsqueeze(1)
    #labels=torch.tensor(labels)
    logits=classifier(pooled_output)
    logits=logits.reshape(1,8)#추가한 코드
    labels=labels.reshape(1)
    #print(logits)
    #print(labels.shape,logits.shape)
    #loss_fct=torch.nn.BCEWithLogitsLoss()
    #loss=loss_fct(logits,labels) #여기서 오류 발생
    loss=loss_fct(logits,labels)
    print('labels-reshape:',labels)
    print('logits:',logits)
    print('loss:',loss)

# for i in range(0,101):
#     labels=aihub_data.iloc[i,10]
#     print(labels)
#     print(type(labels))
#print(aihub_data.iloc[1,10])
#print(type(aihub_data.iloc[1,10])) #numpy.int64

