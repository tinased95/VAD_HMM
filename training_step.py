import numpy as np
from tqdm import tqdm
import pickle
import os
from hmmlearn import hmm
from sklearn.model_selection import train_test_split
from python_speech_features import mfcc, logfbank, delta


def repeatingNumbers(numList):
    indices = []
    i = 0
    while i < len(numList) - 1:
        n = numList[i]
        startIndex = i
        while i < len(numList) - 1 and numList[i] == numList[i + 1]:
            i = i + 1
        endIndex = i
        # print("{0} >> {1}".format(n, [startIndex, endIndex]))
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


def iter_from_X_lengths(X, lengths):
    if lengths is None:
        yield 0, len(X)
    else:
        n_samples = X.shape[0]
        end = np.cumsum(lengths).astype(np.int32)
        start = end - lengths
        if end[-1] > n_samples:
            raise ValueError("more than {:d} samples in lengths array {!s}"
                             .format(n_samples, lengths))

        for i in range(len(lengths)):
            yield start[i], end[i]


def train_model(data):
    learned_hmm = dict()
    for label in data.keys():  # for 0, 1
        print("training label:", label)
        # GaussianHMM
        model = hmm.GMMHMM(n_components=2, covariance_type="diag", n_mix=2)
        length = []
        feat = np.asarray(data[label])
        feature = feat[0]
        length.append(feature.shape[0])
        for f in tqdm(feat[1:]):
            feature = np.concatenate((feature, f), axis=0)
            length.append(f.shape[0])
        obj = model.fit(feature, length)
        print("trained!")
        learned_hmm[label] = obj
    return learned_hmm


def append_delta_features(x_feats):
    x_n_new = []
    for feat in x_feats:
        delta_feat = delta(feat, N=1)
        delta2_feat = delta(delta_feat, N=1)
        feat_39 = np.concatenate((delta_feat, delta2_feat, feat), axis=1)
        x_n_new.append(feat_39)
    return np.asarray(x_n_new)


def main(xypath, outputpath, coeff):
    x = np.load(xypath + 'x.npy')
    y = np.load(xypath + 'y.npy')
    patient_ids = np.load(xypath + 'patient_ids.npy')

    patient_ranges = repeatingNumbers(patient_ids)

    for k in tqdm(range(len(patient_ranges))):  # range(len(patient_ranges)) # TODO
        index_start, index_end, p_id = patient_ranges[k]
        print(index_start, index_end, p_id)

        X_train = np.concatenate((x[:index_start], x[index_end:]), axis=0)
        y_train = np.concatenate((y[:index_start], y[index_end:]), axis=0)

        X_test = x[index_start: index_end]
        y_test = y[index_start: index_end]

        print(X_train.shape, X_test.shape)
        print(y_train.shape, y_test.shape)

        x_n, x_s = [], []
        tedad = repeatingNumbers(y_train)
        for row in tedad:
            if row[0] >= row[1]:
                print("error")
                continue
            if row[2] == 0:
                x_n.append(X_train[row[0]: row[1]].squeeze(axis=1))
            elif row[2] == 1:
                x_s.append(X_train[row[0]: row[1]].squeeze(axis=1))

        if coeff == 39:
            # append delta and double delta features
            x_n = append_delta_features(x_n)
            x_s = append_delta_features(x_s)
            ########################################

        data = dict()
        data[0] = x_n
        data[1] = x_s
        # print(x_feats_for_n)
        print("I am Learning")

        learned_hmm = train_model(data)

        pickle_name = outputpath + "learned" + p_id + ".pkl"
        with open(pickle_name, "wb") as file:
            pickle.dump(learned_hmm, file)
        print("Model Learned")


## xypath, outputpath (where to put trained models)

# 'GaussianHMM_4_states/', 'GaussianHMM_2_states/', 'GMMHMM_8_states_4_mix/', 'GMMHMM_8_states_2_mix/',
#  'GMMHMM_2_states_8_mix/', 'GMMHMM_4_states_8_mix/',
pathss = ['GMMHMM_2_states_4_mix/', 'GMMHMM_4_states_2_mix/']

for pat in pathss:
    main('/scratch/tina/python_speech_features/coeff39/', '/scratch/tina/python_speech_features/coeff39/'+pat, coeff=39)

