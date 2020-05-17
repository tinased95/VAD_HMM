from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
from python_speech_features import mfcc, logfbank, delta
from sklearn.metrics import precision_recall_curve, auc, roc_curve, roc_auc_score, precision_score, recall_score, \
    f1_score
import matplotlib.pyplot as plt
from numpy import interp
from statistics import mean
import matplotlib
import os

matplotlib.use('Agg')


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


def get_windows(y_test, label, win_len=10):
    indices = [i for i, x in enumerate(y_test) if x == label]
    ranges = sum((list(t) for t in zip(indices, indices[1:]) if t[0] + 1 != t[1]), [])
    iranges = iter(indices[0:1] + ranges + indices[-1:])
    range_list = []
    for n in iranges:
        s = n
        e = next(iranges)
        diff = e - s
        number_of_windows = diff // win_len
        for num in range(0, number_of_windows):  # +1 added
            range_list.append([s + (win_len * num), s + (win_len * (num + 1)), label])
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


def calculate_x_y_tests(y_new, X_test, coeff):
    x_test_new = []
    y_test_new = []
    if coeff == 13:
        print("coeff 13 .....")
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


def evaluation_PR(y_true, y_predicted, txt):
    precisions, recalls, thresh = precision_recall_curve(y_true, y_predicted)
    plt.plot(recalls, precisions, marker='.')
    myauc = auc(recalls, precisions)
    print('PR AUC=%.3f' % (myauc))
    plt.title(txt)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()


def evaluation_ROC(y_true, y_predicted, txt):
    myauc = roc_auc_score(y_true, y_predicted)
    print('ROC AUC=%.3f' % (myauc))
    fpr, tpr, _ = roc_curve(y_true, y_predicted)
    plt.plot(fpr, tpr, marker='.', label='ROC')
    plt.title(txt)
    plt.show()


def get_metric_and_best_threshold_from_roc_curve(tpr, fpr, thresholds, num_pos_class, num_neg_class):
    tp = tpr * num_pos_class
    tn = (1 - fpr) * num_neg_class
    acc = (tp + tn) / (num_pos_class + num_neg_class)

    best_threshold = thresholds[np.argmax(acc)]
    return np.amax(acc), best_threshold


def get_metric_and_best_threshold_from_pr_curve(precision, recall, thresholds, num_pos_class, num_neg_class):
    tp = recall * num_pos_class
    fp = (tp / precision) - tp
    tn = num_neg_class - fp
    acc = (tp + tn) / (num_pos_class + num_neg_class)

    best_threshold = thresholds[np.argmax(acc)]
    return np.amax(acc), best_threshold


def main(xypath, trainedpath, coeff, win_len=10):
    print(trainedpath)
    results = []
    results.append("Window length: %d \n" % win_len)
    x = np.load(xypath + 'x.npy')
    y = np.load(xypath + 'y.npy')
    patient_ids = np.load(xypath + 'patient_ids.npy')

    # x.tolist()
    # y.tolist()
    print("xy loaded!")
    patient_ranges = repeatingNumbers(patient_ids)
    print("patient ranges loadded")
    print(len(patient_ranges))
    acc = []
    y_real = []
    y_proba = []
    # PR
    precision_array = []
    recall_array = np.linspace(0, 1, 100)
    # ROC
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    print("here")
    f, axes = plt.subplots(1, 2, figsize=(10, 5))
    print("here")
    for k in tqdm(range(0, len(patient_ranges))):  # ,, len(patient_ranges)
        index_start, index_end, p_id = patient_ranges[k]
        print("----------------------------------------------------- Fold #", k,
              "----------------------------------------------------")

        X_test = x[index_start: index_end]
        y_test = y[index_start: index_end]

        with open(trainedpath + "learned" + p_id + ".pkl", "rb") as file:
            learned_hmm = pickle.load(file)

        tedad = repeatingNumbers(y_test)
        # win_len = 10
        y_new = []
        for row in tedad:
            diff = row[1] - row[0]
            number_of_windows = diff // win_len
            for num in range(0, number_of_windows):  # +1 added
                y_new.append([row[0] + (win_len * num), row[0] + (win_len * (num + 1)), row[2]])

        x_test_new, y_test_new = calculate_x_y_tests(y_new, X_test, coeff)

        y_pred, y_probs, y_max = prediction(x_test_new, learned_hmm)

        # precision recall:
        precision, recall, thresholds = precision_recall_curve(y_test_new, y_probs)
        lab = 'Fold %d AUC=%.4f' % (k + 1, auc(recall, precision))
        #     precision_array = interp(recall_array, recall, precision)
        axes[0].step(recall, precision, label=lab)

        # ROC:
        fpr, tpr, _ = roc_curve(y_test_new, y_probs)
        roc_auc = auc(fpr, tpr)
        mean_tpr = interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        tprs.append(mean_tpr)
        aucs.append(roc_auc)
        lab = 'Fold %d AUC=%.4f' % (k + 1, roc_auc)
        axes[1].step(fpr, tpr, label=lab)


        f1scores = 2 * (precision * recall) / (precision + recall)

        ix = np.argmax(f1scores)
        print('Best Threshold=%f, G-Mean=%.3f \n' % (thresholds[ix], gmeans[ix]))
        print("y_probs:", min(y_probs), max(y_probs), mean(y_probs))
        y_predicted_threshold = np.where(y_probs >= thresholds[ix], 1, 0)
        best_precison = precision_score(y_test_new, y_predicted_threshold)
        best_recall = recall_score(y_test_new, y_predicted_threshold)
        best_f1_score = f1_score(y_test_new, y_predicted_threshold)
        print('Best Precison=%.3f, Best Recall=%.3f, f-1 score=%.3f \n' % (best_precison, best_recall, best_f1_score))
        #################

        y_real.append(y_test_new)
        y_proba.append(y_probs)

        # report(y_test_new, y_predicted_threshold, show_cm=True)
        acc.append(accuracy_score(y_test_new, y_predicted_threshold))
        results.append("--------------------------------- Fold #: %d ------------------------------------ \n" % (k + 1))
        results.append('y probabilities: min= %.3f, max=%.3f, mean=%.3f' % (min(y_probs), max(y_probs), mean(y_probs)))
        results.append('Best Threshold=%f, G-Mean=%.3f \n' % (thresholds[ix], gmeans[ix]))
        results.append(
            'Best Precison=%.3f, Best Recall=%.3f, f-1 score=%.3f \n' % (best_precison, best_recall, best_f1_score))
        results.append(classification_report(y_test_new, y_predicted_threshold, target_names=['nonspeech', 'speech']))
        results.append('\n')

    print(acc)
    results.append(str(acc))
    fout = open(trainedpath + "training_results.txt", 'w')
    for line in results:
        fout.write(line)

    y_real = np.concatenate(y_real)
    y_proba = np.concatenate(y_proba)
    precision, recall, _ = precision_recall_curve(y_real, y_proba)
    np.savez(trainedpath + 'precison_recalls.npz', precision, recall)

    lab = 'Overall AUC=%.4f' % (auc(recall, precision))
    axes[0].step(recall, precision, label=lab, lw=2, color='black')
    axes[0].set_xlabel('Recall')
    axes[0].set_ylabel('Precision')
    axes[0].legend(loc='lower left', fontsize='small')
    axes[0].set_title("PR curve")

    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True positive Rate')
    axes[1].legend(loc='lower right', fontsize='small')
    axes[1].set_title("ROC curve")

    f.tight_layout()
    # plt.show()
    plt.savefig(trainedpath + 'pr_roc_curves.png')


### 1 ---> all files:
# subfolders = [f.path for f in os.scandir('/scratch/tina/python_speech_features/coeff13/') if f.is_dir()]
# for pat in subfolders:
#     main('/scratch/tina/python_speech_features/coeff13/', pat + '/', coeff=13, win_len=10)
### <---

### 2 --- > this is for Gaussians
# nofcomponents = [2, 4, 8, 10, 12]
# for n in nofcomponents:
#     filename = 'GaussianHMM_' + str(n) + '_states/'
#     print(filename)
#     main('/scratch/tina/python_speech_features/coeff13/', '/scratch/tina/python_speech_features/coeff13/' + filename, coeff=13, win_len=10)
#
#
# print("##################### finished coeff 13 #####################")
# nofcomponents = [2, 4, 8, 10, 12]
# for n in nofcomponents:
#     filename = 'GaussianHMM_' + str(n) + '_states/'
#     print(filename)
#     main('/scratch/tina/python_speech_features/coeff13/', '/scratch/tina/python_speech_features/coeff39/' + filename, coeff=39, win_len=10)
### <---

### 3 ---> this is for coeff13 GMMHMMs
# nofcomponents = [8, 10]  # if you add , 12 it will be a complete package
# nofcomponents2 = [2, 4, 8, 10, 12]
# for n in nofcomponents:
#     for m in nofcomponents2:
#         filename = 'GMMHMM_' + str(n) + '_states_' + str(m) + '_mix/'
#         print(filename)
#         main('/scratch/tina/python_speech_features/coeff13/', '/scratch/tina/python_speech_features/coeff13/' + filename, coeff=13, win_len=10)
### <---

### 4  ---> testing the ones ready in coeff39
# [12,12], [12,10], [12,2], [2,10], [2,12], [4,10], [4,12], [8,10],[8,12], [12,4],
lists = [[12, 8]]
# lists=[[10,10], [10,12], [10,2], [10,4], [10,8], [2,2], [2,4], [2,8], [4,2], [4,4], [4,8],
#        [8,2], [8,4], [8,8]]

for l in lists:
    filename = 'GMMHMM_' + str(l[0]) + '_states_' + str(l[1]) + '_mix/'
    print(filename)
    print("Starting: ", filename)
    main('/scratch/tina/python_speech_features/coeff13/', '/scratch/tina/python_speech_features/coeff39/' + filename,
         coeff=39, win_len=10)

### <---


### remaining::: for the morning :)
#