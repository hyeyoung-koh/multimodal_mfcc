#모든 파일에 대해 mp4->mp3->wav를 해보자.

import moviepy.editor as mp
import os
import pandas as pd
from pydub import AudioSegment

def loop_directory(directory:str):
    for filename in os.listdir(directory): #특정 디렉토리의 파일을 반복한다.
        if filename.endswith('.wav'):


#  여러 mp3파일을 wav로 변환
audio_files=os.listdir('path')
len_audio=len(audio_files)
for i in range(len_audio):
    if os.path.splitext(audio_files[i])[1]=='mp4':
        clip=mp.VideoFileClip(audio_files[i])
        clip.audio.write_audiofile('clip_'+i+'.mp3') #mp3로 바꿈.
        clip_mp3file=AudioSegment.from_mp3('clip_'+i+'mp3')#mp3파일을 wav로 바꿈.
        src='clip_'+i+'.mp3'
        dst='clip_'+i+'.wav'
        sound=AudioSegment.from_mp3(src)
        sound.export(dst,format='wav')
    else:
        continue