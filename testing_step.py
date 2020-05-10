from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle


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


def get_windows(y_test, label, win_len = 10):
    indices = [i for i, x in enumerate(y_test) if x ==  label]
    ranges = sum((list(t) for t in zip(indices, indices[1:]) if t[0]+1 != t[1]), [])
    iranges = iter(indices[0:1] + ranges + indices[-1:])
    range_list = []
    for n in iranges:
        s = n
        e = next(iranges)
        diff = e - s
        number_of_windows = diff // win_len
        for num in range (0,number_of_windows): # +1 added
            range_list.append([s + (win_len * num) , s + (win_len * (num+1)) , label])
    return range_list


def prediction(test_data, trained):
    # predict list of test
    predict_label = []
    predict_probs = []
    predict_max = []
    for test in tqdm(test_data):
        scores = []
        for node in trained.keys():
            scores.append(trained[node].score(test))
        predict_label.append(scores.index(max(scores)))
        predict_probs.append(scores[1] - scores[0])
        predict_max.append(max(scores))

    return predict_label, predict_probs, predict_max


def report(y_test, y_pred, show_cm=True):
#     print("confusion_matrix:\n\n", confusion_matrix(y_test, y_pred))
    print("----------------------------------------------------------")
    print("----------------------------------------------------------\n")
    print("Classification based on outputting the most likely class by finding the maximum between two scores")
    print("\n")
    print("classification_report:\n\n", classification_report(y_test, y_pred, target_names=['nonspeech', 'speech']))
#     print("----------------------------------------------------------")
    print("----------------------------------------------------------\n")
    print("Accuracy:", accuracy_score(y_test, y_pred))
#     print("----------------------------------------------------------")
    print("----------------------------------------------------------\n")
    # if show_cm:
        # plot_confusion_matrix(confusion_matrix(y_test, y_pred), ['nonspeech', 'speech'])


def main(xypath, trainedpath):
    x = np.load(xypath + 'x.npy')
    y = np.load(xypath + 'y.npy')
    patient_ids = np.load(xypath + 'patient_ids.npy')

    # x.tolist()
    # y.tolist()

    patient_ranges = repeatingNumbers(patient_ids)

    acc = []
    for k in tqdm(range(len(patient_ranges))): # ,, len(patient_ranges)

        index_start, index_end, p_id = patient_ranges[k]
        print("----------------------------------------------------- Fold #", k,
              "----------------------------------------------------")

        X_test = x[index_start: index_end]
        y_test = y[index_start: index_end]

        with open(trainedpath + "learned" + p_id + ".pkl", "rb") as file:
            learned_hmm = pickle.load(file)

        # range_list_1 = get_windows(y_test, 1, win_len=2000)
        # range_list_0 = get_windows(y_test, 0, win_len=2000)
        #
        # y_new = sorted(range_list_0 + range_list_1)  # windowed y labels and ranges
        tedad = repeatingNumbers(y_test)
        win_len = 10
        y_new = []
        for row in tedad:
            diff = row[1] - row[0]
            number_of_windows = diff // win_len
            for num in range(0, number_of_windows):  # +1 added
                y_new.append([row[0] + (win_len * num), row[0] + (win_len * (num + 1)), row[2]])


        # x_test_new = []
        # y_test_new = []
        # for s_e_l in y_new:
        #     lists = []
        #     for se in range(s_e_l[0], s_e_l[1]):
        #         lists.append(X_test[se].flatten())
        #     lists = np.asarray(lists)
        #     x_test_new.append(lists)
        #     y_test_new.append(s_e_l[2])
        # x_test_new = np.asarray(x_test_new)
        # y_test_new = np.asarray(y_test_new)

        x_test_new = []
        y_test_new = []
        for s_e_l in y_new:
            x_test_new.append(X_test[s_e_l[0]: s_e_l[1]].squeeze(axis=1))
            y_test_new.append(s_e_l[2])

        x_test_new = np.asarray(x_test_new)
        y_test_new = np.asarray(y_test_new)

        y_pred, y_probs, y_max = prediction(x_test_new, learned_hmm)

        print("Classification based on logp1 - logp2")
        print("\n")

        report(y_test_new, y_pred, show_cm=True)
        acc.append(accuracy_score(y_test_new, y_pred))

    print(acc)


main('python_speech_features/coeff13/', 'python_speech_features/coeff13/GMMHMM_8_states/')