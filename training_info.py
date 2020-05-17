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

def calculate_x_y_tests(y_new, X_test, coeff):
    x_test_new = []
    y_test_new = []
    if coeff == 13:
        # print("coeff 13 .....")
        for s_e_l in y_new:
            x_test_new.append(X_test[s_e_l[0]: s_e_l[1]].squeeze(axis=1))
            y_test_new.append(s_e_l[2])
        return np.asarray(x_test_new), np.asarray(y_test_new)
    elif coeff == 39:
        for s_e_l in y_new:
            feat = X_test[s_e_l[0]: s_e_l[1]].squeeze(axis=1)
            delta_feat = delta(feat, N=1)
            delta2_feat = delta(delta_feat, N=1)
            feat_39 = np.concatenate((delta_feat, delta2_feat, feat), axis=1)
            x_test_new.append(feat_39)
            y_test_new.append(s_e_l[2])
        return np.asarray(x_test_new), np.asarray(y_test_new)
    else:
        "Not defined!"


def main(xypath, outputpath, coeff):
    win_len = 10
    x = np.load(xypath + 'x.npy')
    y = np.load(xypath + 'y.npy')
    patient_ids = np.load(xypath + 'patient_ids.npy')

    patient_ranges = repeatingNumbers(patient_ids)

    for k in tqdm(range(len(patient_ranges))):  # range(len(patient_ranges)) # TODO
        index_start, index_end, p_id = patient_ranges[k]
        print("Indexes: ", index_start, index_end, p_id)

        X_train = np.concatenate((x[:index_start], x[index_end:]), axis=0)
        y_train = np.concatenate((y[:index_start], y[index_end:]), axis=0)

        X_test = x[index_start: index_end]
        y_test = y[index_start: index_end]

        # print("X_train:", X_train.shape, "y_train:", y_train.shape)
        # print("X_test:", X_test.shape, "y_test", y_test.shape)

        x_n, x_s = [], []
        tedad = repeatingNumbers(y_train)
        for row in tedad:
            if row[0] >= row[1]:
                # print("error")
                continue
            if row[2] == 0:
                x_n.append(X_train[row[0]: row[1]].squeeze(axis=1))
            elif row[2] == 1:
                x_s.append(X_train[row[0]: row[1]].squeeze(axis=1))

        ns = []
        for seq in x_n:  # each seq is a (nx13) array
            ns.append(np.asarray(seq).shape[0])

        ss = []
        for seq in x_s:  # each seq is a (nx13) array
            ss.append(np.asarray(seq).shape[0])
        print("number of non speech sequences:", len(x_n))
        print("number of speech sequences:", len(x_s))
        print("sum ns: ", sum(ns))
        print("sum ss:", sum(ss))

        print(min(ss), max(ss), sum(ss)/len(ss))
        tedad = repeatingNumbers(y_test)
        # win_len = 10
        y_new = []
        for row in tedad:
            diff = row[1] - row[0]
            number_of_windows = diff // win_len
            for num in range(0, number_of_windows):  # +1 added
                y_new.append([row[0] + (win_len * num), row[0] + (win_len * (num + 1)), row[2]])

        x_test_new, y_test_new = calculate_x_y_tests(y_new, X_test, 13)
        # print(y_test_new.shape)


main('/scratch/tina/python_speech_features/coeff13/', '/scratch/tina/python_speech_features/coeff13/GaussianHMM_4_states', coeff=13)

