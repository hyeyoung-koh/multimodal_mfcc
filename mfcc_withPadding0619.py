import librosa
import torch
import sklearn
import numpy as np


def mfcc_feature(file):
    y, sr = librosa.load(file, sr=None)  # sampling rate
    mfcc_extracted = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    print('myfile명:', file)
    mfcc_extracted = torch.from_numpy(mfcc_extracted).float()

    # scaling & padding
    mfcc_extracted_reshape = mfcc_extracted.view(1, -1)
    mfcc_scale = sklearn.preprocessing.scale(mfcc_extracted_reshape, axis=1)

    pad2d = lambda a, i: a[:, 0:i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0], i - a.shape[1]))))
    padded_mfcc = pad2d(mfcc_scale, 6000)
    padded_mfcc = torch.Tensor(padded_mfcc)
    return padded_mfcc
