import os
import glob

from pydub import AudioSegment


def convert_from_mp4(file_path):  #  mp4->wav
    try:
        track=AudioSegment.from_file(file_path, 'mp4')
        file_handle=track.export(file_path.replace('.mp4', '.wav'), format='wav')
    except FileNotFoundError:
        print("{} does not appear to be valid. Please check.")
    except Exception as e:
        print(e)


def convert_files_to_wav(dir_path, audio_format="m4a"):
    files=glob.glob(os.path.join(dir_path, "*."+audio_format))
    if audio_format=='mp4':
        cntr = 0
        for aud_file in files:
            convert_from_mp4(aud_file)
            cntr += 1
        print("{} {} files converted to .wav and saved in {}".format(cntr, audio_format, dir_path))


i=0
for i in range(1907, 1985): #1992까지 저장함.
    myfile = 'clip_'+str(i)
    convert_file = convert_files_to_wav('F:\\1601-2000\\1601-2000\\'+myfile, 'mp4')
    i+=1

