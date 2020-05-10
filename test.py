import librosa
from python_speech_features import mfcc, logfbank, delta
import numpy as np

y, sr = librosa.load(librosa.util.example_audio_file())
# mfcc_lib = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
# mfcc_delta = librosa.feature.delta(mfcc_lib)
# mfcc_delta2 = librosa.feature.delta(mfcc_lib, order=2)

mfcc_features = mfcc(y, sr)
deltaaa = delta(mfcc_features, N=1)
deltaaa2 = delta(deltaaa, N=1)
#
print(mfcc_features.shape)
print(deltaaa.shape)
print(deltaaa2.shape)
# print(deltaaa.shape, deltaaa2.shape)

print(np.concatenate((deltaaa, deltaaa2, mfcc_features), axis=1).shape)


# x = np.load('python_speech_features/coeff39//x.npy')
# y = np.load('python_speech_features/coeff39//y.npy')
# patient_ids = np.load('python_speech_features/coeff39//patient_ids.npy')
#
# print(x.shape)