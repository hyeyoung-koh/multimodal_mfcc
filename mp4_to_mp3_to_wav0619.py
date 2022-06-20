import moviepy.editor as mp
from pydub import AudioSegment

clip = mp.VideoFileClip('clip_1.mp4') #  오디오를 추출하는 기능을 한다.
clip.audio.write_audiofile('clip_1.mp3') #  clip_1.mp3파일이 만들어짐.
src='clip_1.mp3'
dst='clip_1.wav'
sound=AudioSegment.from_mp3(src)
sound.export(dst, format='wav')
