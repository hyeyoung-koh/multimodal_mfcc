#  MFCC는 음성 데이터를 특징벡터화해주는 알고리즘으로, 아날로그 신호인 음성데이터를 벡터화함으로써 학습데이터로 사용할 수 있게 된다.
import os
import librosa

#  filepath='C:\Users\hyeyoung\PycharmProjects\mfcc\clip_1.wav'
#  signal,sr=librosa.load(filepath,sr=16000)
#  sr은 음성 데이터의 형식에 따라 다른데,.wav파일의 경우는 16000이다.
#  MFCCs=librosa.feature.mfcc(signal,sr,n_fft=400,hop_length=160,n_mfcc=36)
#  sr는 default가 22050Hz이며 앞에서 16000Hz로 받았으므로 sr값을 동일하게 설정해준다.
#  n_fft는 음성의 길이를 얼만큼으로 자를지에 대한 파라미터이다.
#  자연어처리에서는 frame length를 25ms를 기본으로 한다.n_fft를 400으로 설정하게 되면 frame length가 0.025(n_fft/sr),즉,25ms가 된다.
#  hop_length는 그 길이만큼 데이터를 읽어간다.frame stride=10ms가 default이므로 sr*frame_stride=160을 통해 hop_length를 160으로 설정해준다.
#  n_mfcc:기본이 20이다. 특징 벡터의 개수를 정해주는 부분이다. 여기서는 svm테스트해보니까 36과 50이 높게 나옴->36으로 결정.
#  n_mfcc값이 커질수록 더 많은 특징벡터를 추출하게 된다.


def getMFCC(i):
    signal,sr=librosa.load(i, sr=16000)
    #  MFCC를 통한 특징 벡터 추출(n_mfcc=36)
    MFCCs = librosa.feature.mfcc(signal, sr, n_fft=400, hop_length=160, n_mfcc=36)
    return MFCCs


f=getMFCC('clip_1.wav')
print(f)  # mfcc로 변환됨