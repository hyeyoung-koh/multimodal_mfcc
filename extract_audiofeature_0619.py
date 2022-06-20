from pydub import AudioSegment
import pandas as pd
import os


def extract_audio_feature(wav_file):
    audio=AudioSegment.from_wav(wav_file)
    audio=audio.set_channels(1)
    audio=audio.get_array_of_samples()
    return audio


df1 = pd.read_csv('/home/iai/PycharmProjects/extractMFCC/man5000_2.csv',encoding='utf-8-sig')

