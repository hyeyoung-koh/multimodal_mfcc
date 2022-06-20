#import moviepy
import moviepy.editor as mp

#define the video clip
my_clip=mp.VideoFileClip(r"videotest.mov")
my_clip

#extracting the audio
#video format은 WMV,OGG,3GP,MP4이며 audio format은 MP3,AAC,WMA,AC3이다.
my_clip.audio.write_audiofile(r"my_result.mp3")

#이와 같은 방법인게
from moviepy.editor import *
mp4_file='video.mp4'
mp3_file='audio.mp3'

videoClip=VideoFileClip(mp4_file)
audioclip=videoClip.audio

