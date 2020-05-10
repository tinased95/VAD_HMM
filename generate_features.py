import librosa
import librosa.display
from python_speech_features import mfcc, logfbank, delta
import numpy as np
import pandas as pd
from tqdm import tqdm

LABELS_PATH = '/mnt/FS1/copd-data/speech/mapped_labels.csv'
RELATIVE_PATH = '/mnt/FS1/copd-data/main/features/'
sampling_rate = 16000
FRAME_LEN = 0.025 * sampling_rate
FRAME_STEP = 0.01 * sampling_rate


def feature_extractor(audio, mode='librosa'):
    if mode == 'librosa':
        mfcc_features = librosa.feature.mfcc(y=audio, sr=sampling_rate, n_mfcc=13)
        return mfcc_features.T  # (1, 13)
    else:
        mfcc_features = mfcc(audio, sampling_rate)  # (1, 13)
        # deltaaa = delta(mfcc_features, N=1)
        # deltaaa2 = delta(mfcc_features, N=2)
        # return np.concatenate((mfcc_features, deltaaa, deltaaa2), axis=1)
        return mfcc_features


def append_non_speech_labels(df, filename): # this function will be executed for each audio file
    starts = df['src_start_ts'].tolist() # all the starting times
    ends = df['src_end_ts'].tolist() # all the ending times
    num = len(starts) # how many rows
    missing_starts, missing_ends, duration = [], [], []
    for i in range(0, num-1):
        missing_starts.append(ends[i])
        missing_ends.append(starts[i+1])
        duration.append(starts[i+1]-ends[i] )

    newdf = pd.DataFrame()
    newdf['src_start_ts'] = missing_starts
    newdf['src_end_ts'] = missing_ends
    newdf['duration'] = duration
    newdf['label'] = 'nonspeech'
    newdf['src_file'] = filename
    newdf = newdf[newdf['duration'] > 0]  # two rows have wrong labels!

    result = pd.concat([newdf, df])
    return result.sort_values(by='src_start_ts')


def read_labels():
    COL_LIST = ["src_start_ts", "src_end_ts", "label", "src_file"]
    df = pd.read_csv(LABELS_PATH, usecols=COL_LIST)  # TODO remove nrows=10

    df['duration'] = df['src_end_ts'] - df['src_start_ts']
    df = df[df['duration'] > 0]  # two rows have wrong labels!
    return df.groupby(['src_file'])


def generate_features(grp, mode, outputpath):
    """
    grp: preprocess dataframe (grouped by filenames)
    mode: librosa or python_speech_features
    outputpath: where to save the x, y, patient_ids arrays
    """
    x = []
    y = []
    patient_ids = []
    for filename, group in tqdm(grp):
        if group.shape[0] <= 2:
            continue

        partial_df = append_non_speech_labels(group, filename)
        partial_df.reset_index(inplace=True, drop=True)  # TODO this line added newly to reset index

        wav, sr = librosa.load(RELATIVE_PATH + str(filename), sr=None)
        # wav = librosa.resample(wav, sr, sampling_rate) # TODO don't need this line since its already 16000

        for index, row in partial_df[['src_start_ts', 'src_end_ts', 'label']].iterrows():

            s, e, label = row[0], row[1], row[2]
            number = int(((e - s) - FRAME_LEN) // FRAME_STEP)
            for i in range(number):
                feature = feature_extractor(wav[int(i * FRAME_STEP + s): int((i * FRAME_STEP + s) + FRAME_LEN)], mode) # mode='else'
                x.append(feature)
                y.append(0 if label == 'nonspeech' else 1)
                patient_ids.append(filename[11:13])

    np.save(outputpath + 'x.npy', x)
    np.save(outputpath + 'y.npy', y)
    np.save(outputpath + 'patient_ids.npy', patient_ids)


grp = read_labels()
generate_features(grp, 'else', 'python_speech_features/coeff39/')
