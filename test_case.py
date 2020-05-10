import librosa
import librosa.display
from python_speech_features import mfcc, logfbank, delta
import numpy as np
import pandas as pd
from tqdm import tqdm
import numpy as np
from tqdm import tqdm
import pickle
import os
from hmmlearn import hmm

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
        deltaaa = delta(mfcc_features, N=1)
        deltaaa2 = delta(mfcc_features, N=2)
        return np.concatenate((mfcc_features, deltaaa, deltaaa2), axis=1)


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
    df = pd.read_csv(LABELS_PATH, usecols=COL_LIST, nrows=50)  # TODO remove nrows=10
    # print(df.shape)
    df['duration'] = df['src_end_ts'] - df['src_start_ts']
    df = df[df['duration'] > 0]  # two rows have wrong labels!
    return df.groupby(['src_file'])



def repeatingNumbers(numList):
    indices = []
    i = 0
    while i < len(numList) - 1:
        n = numList[i]
        startIndex = i
        while i < len(numList) - 1 and numList[i] == numList[i + 1]:
            i = i + 1
        endIndex = i
        print("{0} >> {1}".format(n, [startIndex, endIndex]))
        indices.append([startIndex, endIndex, n])
        i = i + 1
    return indices


def create_sequences(y_train, x_train, label):
    '''
    inputs: y_train, x_train
    outputs: new x_train (windowed)

    create sequences based on continous lables in y_train:
    for example:
    y_train = [1,1,1,1,0,0,1,1,1,0]
    --> new y_train = [1,0,1,0]

    X_train = [(1,13), (1,13), (1,13), (1,13), (1,13), (1,13), (1,13), (1,13), (1,13), (1,13)]
    --> new X_train = [(4,13), (2,13), (3,13), (1,13)]
    '''
    indices = [i for i, x in enumerate(y_train) if x == label]  # get all indices of 1 in y_train
    ranges = sum((list(t) for t in zip(indices, indices[1:]) if t[0] + 1 != t[1]), [])
    iranges = iter(indices[0:1] + ranges + indices[-1:])
    range_list = []
    for n in iranges:
        range_list.append([n, next(iranges)])

    x_feats = []
    for se in range_list:
        sequence = []
        for i in range(se[0], se[1] + 1):
            sequence.append(np.asarray(x_train[i]).flatten())  # (1,13) --> (13,)
        sequence = np.asarray(sequence)
        x_feats.append(sequence)  # (?,13)

    return np.asarray(x_feats)  # (n, ?, 13)


def train_model(data):
    learned_hmm = dict()
    for label in data.keys(): # for 0, 1
        print("training label:", label)
        model = hmm.GMMHMM(n_components=4)
        length = []
        feat = np.asarray(data[label])
        feature = feat[0]
        length.append(feature.shape[0])
        for f in tqdm(feat[1:]):
            feature = np.concatenate((feature, f),axis=0)
            length.append(f.shape[0])
#             print(f.shape)
        obj = model.fit(feature, length)
        print("trained!")
        learned_hmm[label] = obj
    return learned_hmm




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
        # print("here")
        if group.shape[0] <= 2:
            continue

        partial_df = append_non_speech_labels(group, filename)

        wav, sr = librosa.load(RELATIVE_PATH + str(filename), sr=None)
        wav = librosa.resample(wav, sr, sampling_rate)

        for index, row in partial_df[['src_start_ts', 'src_end_ts', 'label']].iterrows():

            s, e, label = row[0], row[1], row[2]
            print(index)
            number = int(((e - s) - FRAME_LEN) // FRAME_STEP)
            for i in range(number):
                feature = feature_extractor(wav[int(i * FRAME_STEP + s): int((i * FRAME_STEP + s) + FRAME_LEN)], mode) # mode='else'
                x.append(feature)
                y.append(0 if label == 'nonspeech' else 1)
                patient_ids.append(filename[11:13])

    np.save(outputpath + 'x.npy', x)
    np.save(outputpath + 'y.npy', y)
    np.save(outputpath + 'patient_ids.npy', patient_ids)


def training_step(mypath):
    x = np.load(mypath + 'x.npy')
    y = np.load(mypath + 'y.npy')

    patient_ids = np.load(mypath + 'patient_ids.npy')

    x = x.tolist()
    y = y.tolist()

    patient_ranges = repeatingNumbers(patient_ids)
    y_pred, y_probs, y_max, y_true = [], [], [], []
    for k in tqdm(range(0, len(patient_ranges))):  # range(len(patient_ranges)) # TODO
        index_start, index_end, p_id = patient_ranges[k]
        print(index_start, index_end, p_id)
        # X_test = x[index_start: index_end]
        # y_test = y[index_start: index_end]
        X_train = x[:index_start] + x[index_end:]
        y_train = y[:index_start] + y[index_end:]

        print(min(y_train), max(y_train))

        # x_feats_for_s = []
        # x_feats_for_n = []

        x_feats_for_s = create_sequences(y_train, X_train, 1)
        x_feats_for_n = create_sequences(y_train, X_train, 0)

        print(x_feats_for_n.shape)
        data = dict()
        data[0] = x_feats_for_n
        data[1] = x_feats_for_s
        # print(x_feats_for_n)
        print("I am Learning")

        learned_hmm = train_model(data)

        pickle_name = mypath + "learned" + p_id + ".pkl"
        with open(pickle_name, "wb") as file:
            pickle.dump(learned_hmm, file)
        print("Model Learned")


# 50 rows (10 files)
##### generate_features
grp = read_labels()
generate_features(grp, 'else', 'test/')
#########################

##### training
# training_step('test/')