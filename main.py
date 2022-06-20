pip install python_speech_features

def mfcc(signal,samplerate=16000,winlen=0.025,winstep=0.01,numcep=13,
                 nfilt=26,nfft=512,lowfreq=0,highfreq=None,preemph=0.97,
     ceplifter=22,appendEnergy=True)

#setup.py파일
try:
    from setuptools import setup  # enables develop
except ImportError:
    from distutils.core import setup

setup(name='python_speech_features',
      version='0.6.1',
      description='Python Speech Feature extraction',
      author='James Lyons',
      author_email='james.lyons0@gmail.com',
      license='MIT',
      url='https://github.com/jameslyons/python_speech_features',
      packages=['python_speech_features'],
      install_requires=[
          'numpy',
          'scipy',
      ]
#example.py파일
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav

(rate,sig) = wav.read("english.wav")
mfcc_feat = mfcc(sig,rate)
d_mfcc_feat = delta(mfcc_feat, 2)
fbank_feat = logfbank(sig,rate)

print(fbank_feat[1:3,:])


i=0
for i in range(1201,1601):
    myfile='D:\1201-1600\1201-1600'+clip_'+str(i)
    convert_file=convert_files_to_wav('D:\\1201-1600\\1201-1600\\'+myfile,'mp4')
    i+=1


i=0
for i in range(1601,2001):
    myfile='clip_'+str(i)
    convert_file=convert_files_to_wav('D:\\1601-2000\\1601-2000\\'+myfile,'mp4')
    i+=1

i=0
for i in range(2001,2401):
    myfile='clip_'+str(i)
    convert_file=convert_files_to_wav('D:\\2001-2400\\2001-2400\\'+myfile,'mp4')
    i+=1

i=0
for i in range(2401,2801):
    myfile='clip_'+str(i)
    convert_file=convert_files_to_wav('D:\\2401-2800\\2401-2800\\'+myfile,'mp4')
    i+=1

i=0
for i in range(2801,3201):
    myfile='clip_'+str(i)
    convert_file=convert_files_to_wav('D:\\2801-3200\\2801-3200\\'+myfile,'mp4')
    i+=1

i=0
for i in range(3201,3601):
    myfile='clip_'+str(i)
    convert_file=convert_files_to_wav('D:\\3201-3600\\3201-3600\\'+myfile,'mp4')
    i+=1

i=0
for i in range(3601,4001):
    myfile='clip_'+str(i)
    convert_file=convert_files_to_wav('D:\\3601-4000\\3601-4000\\'+myfile,'mp4')
    i+=1

0

i=0
for i in range(4001,4401):
    myfile='clip_'+str(i)
    convert_file=convert_files_to_wav('D:\\4001-4400\\4001-4400\\'+myfile,'mp4')
    i+=1

i=0
for i in range(4401,4801):
    myfile='clip_'+str(i)
    convert_file=convert_files_to_wav('D:\\4401-4800\\4401-4800-수정본\\'+myfile,'mp4')
    i+=1

i=0
for i in range(4801,5201):
    myfile='clip_'+str(i)
    convert_file=convert_files_to_wav('D:\\4801-5200\\4801-5200\\'+myfile,'mp4')
    i+=1

i=0
for i in range(5201,5601):
    myfile='clip_'+str(i)
    convert_file=convert_files_to_wav('D:\\5201-5600\\5201-5600\\'+myfile,'mp4')
    i+=1
-----

i=0
for i in range(4001,4401):
    myfile='clip_'+str(i)
    convert_file=convert_files_to_wav('D:\\4001-5200\\4001-5200\\'+myfile,'mp4')
    i+=1

i=0
for i in range(4401,4801):
    myfile='clip_'+str(i)
    convert_file=convert_files_to_wav('D:\\4401-5600\\4401-5600-수정본\\'+myfile,'mp4')
    i+=1



#-----------------------------
i=0
for i in range(4801,5201):
    myfile='clip_'+str(i)
    convert_file=convert_files_to_wav('D:\\4801-5200\\4801-5200-수정본\\'+myfile,'mp4')
    i+=1

i=0
for i in range(5201,5601):
    myfile='clip_'+str(i)
    convert_file=convert_files_to_wav('D:\\5201-5600\\5201-5600-수정본\\'+myfile,'mp4')
    i+=1










