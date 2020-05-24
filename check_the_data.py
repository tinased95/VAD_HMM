import librosa
import librosa.display
import pathlib
import itertools
import numpy, scipy, matplotlib.pyplot as plt, sklearn, librosa, urllib
import urllib.request
from python_speech_features import mfcc, logfbank, delta
from hmmlearn import hmm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from itertools import groupby
from operator import itemgetter
import pandas as pd
from tqdm import tqdm
import glob
import pathlib
import pandas as pd
import json

IDs = ['20', '21', '23', '25', '29', '30']


def create_csv_from_json_files():
    for id in IDs:
        with open('/mnt/FS1/copd-data/speech/audio_files/copdpatient' + id + '/noNoise/segments.json') as f:
            data = json.load(f)

        df = pd.DataFrame.from_records(data)
        df.drop(columns=['file'], inplace=True)
        df['source'] = df['source'].str[35:]
        df = df[['start_merged', 'end_merged', 'start_src', 'end_src', 'source']]
        df.to_csv(LABELS_PATH + 'segments' + id + '.csv')


def detect_silence_labels(df, filename): # this function will be executed for each audio file
    starts = df['start_src'].tolist() # all the starting times
    ends = df['end_src'].tolist() # all the ending times
    num = len(starts) # how many rows
    missing_starts, missing_ends, duration = [], [], []
    if starts[0]!=0:
        missing_starts.append(0)
        missing_ends.append(starts[0])
        duration.append(starts[0])
    for i in range(0, num-1):
        missing_starts.append(ends[i])
        missing_ends.append(starts[i+1])
        duration.append(starts[i+1]-ends[i])

    if ends[-1] < 1920000:
        missing_starts.append(ends[-1])
        missing_ends.append(1920000)
        duration.append(1920000 - ends[-1])

    newdf = pd.DataFrame()
    newdf['start'] = missing_starts
    newdf['end'] = missing_ends
    newdf['duration'] = duration
    newdf['label'] = 'silence'
    newdf['src_file'] = filename

    return newdf


def find_silence_between_2_minutes(partial_df, filename):
    df_silenece = detect_silence_labels(partial_df, filename)
    df_silenece.reset_index(inplace=True, drop=True)  # TODO this line added newly to reset index

    return df_silenece

LABELS_PATH = '/home/tina/research/labels_and_maps/'

txtfiles = list(pathlib.Path(LABELS_PATH).glob('*.txt'))
txtfiles.sort()
mapfiles = list(pathlib.Path(LABELS_PATH).glob('*.csv'))
mapfiles.sort()

# f = open(str(txtfiles[0]), 'r')
# lines = f.readlines()
# for line in range(2):
#     l = lines[line].strip().split()
#     start, end = float(l[0]), float(l[1])
#     print(start, end)

df = pd.read_csv(mapfiles[0])
grp = df.groupby(['source'])

i = 0
for filename, group in tqdm(grp):
    if i == 0:
        df_silence = detect_silence_labels(group, filename)
        print(df_silence['duration'].sum()/16000)
    i = 1

# print(grp[0])
# find_silence_between_2_minutes(grp[0])
