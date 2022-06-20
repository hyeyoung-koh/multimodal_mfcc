# moviepy 모듈 설치
#pip install moviepy

# 모듈 로딩 후 오디오 추출
import moviepy.editor as mp
import pydub
from os import path
from pydub import AudioSegment

clip = mp.VideoFileClip("file to extract audio.mp4")
clip.audio.write_audiofile("audio.mp3") #얘는 mp3파일이다.

#  mp3파일을 wav로 변환하기-pydub패키지가 필요하다.
#  pip install pydub

#  ffmpeg설치가 필요하다 ;
#  files
src='transcript.mp3'
dst='test.wav'


#  convert mp3 to wav ->아마도???밑의 코드 참고해서 변형한 것임.
sound=AudioSegment.from_mp3(dst)
sound.export(src, format='mp3')


#  convert wav to mp3
sound=AudioSegment.from_mp3(src)
sound.export(dst, format='wav')
