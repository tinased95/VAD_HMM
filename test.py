import librosa
from python_speech_features import mfcc, logfbank, delta
import numpy as np

import matplotlib.pyplot as plt
import numpy
from sklearn.datasets import make_blobs
from sklearn.metrics import precision_recall_curve, auc
from sklearn.model_selection import KFold
from sklearn.svm import SVC

from PIL import Image
import PIL
import os
import glob
from pathlib import Path
import pickle
# subfolders = [f.path for f in os.scandir('/scratch/tina/python_speech_features/coeff13/') if f.is_dir()]
# subfolders = sorted(subfolders)
#
# for subfolder in subfolders:
#     # creating a image object (main image)
#     img_path = glob.glob(subfolder + '/*.png')[0]
#     print(os.path.basename(img_path))
#     img = Image.open(img_path)
#     img.show()
#     # save a image using extension
#     img = img.save("/scratch/tina/python_speech_features/images/" + os.path.basename(Path(subfolder)) + ".png")


# subfolders.remove('/scratch/tina/python_speech_features/coeff13/GaussianHMM_10_states')
# from shutil import copyfile
# for subfolder in subfolders:
#     print(subfolder)
#     txt_path = glob.glob(subfolder + '/*.txt')[0]
#     copyfile(txt_path, '/scratch/tina/python_speech_features/texts/coeff13/' + os.path.basename(Path(subfolder)) + ".txt")

# GMMHMM_2_states_2_mix
# GaussianHMM_2_states
with open('/scratch/tina/python_speech_features/coeff13/GaussianHMM_4_states/learned30.pkl', "rb") as file:
    learned_hmm = pickle.load(file)


print(learned_hmm[0].transmat_)
    # print(learned_hmm[1].monitor_)
    # print(learned_hmm[1].startprob_)