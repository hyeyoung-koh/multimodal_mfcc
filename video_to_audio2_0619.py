import moviepy.editor as mp
import os
from pydub import AudioSegment


print(os.getcwd())
clip = mp.VideoFileClip('clip_1.mp4')
clip.audio.write_audiofile('clip_1.mp3')


#  mp3파일을 wav 파일로 변환하자.
clip1_mp3file=AudioSegment.from_mp3('clip_1.mp3')
#  clip1_mp3file=AudioSegment.from_mp3(src)
#  print(type(clip1_mp3file))
clip1_mp3file.export(out_f='clip_1.wav', format='wav')
#  dst=clip1_mp3file.export(out_f='clip_1.wav',format='wav')
#  files
src='clip_1.mp3'
dst='clip_1.wav'

sound=AudioSegment.from_mp3(src)
sound.export(dst, format='wav')