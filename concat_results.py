import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import auc
from pathlib import Path
matplotlib.use('Agg')


### 1 ---> all files:
# subfolders = [f.path for f in os.scandir('/scratch/tina/python_speech_features/coeff39/') if f.is_dir()]
# subfolders = sorted(subfolders)
#
# fig = plt.figure(figsize=(10, 10))
# # testt = ['/scratch/tina/python_speech_features/coeff39/GMMHMM_2_states_2_mix/', '/scratch/tina/python_speech_features/coeff39/GMMHMM_4_states_2_mix/']
# for subfolder in subfolders:
#     pr_npz = glob.glob(subfolder + '/*.npz')[0]
#     pr = np.load(pr_npz)
#     precision = pr['arr_0']
#     recall = pr['arr_1']
#     lab = os.path.basename(Path(subfolder)) + ' AUC: ' + str(round(auc(recall, precision),4))
#     plt.step(recall, precision, label=lab)
#
# # plt.show()
# plt.legend(loc='upper right', fontsize='small')
# plt.savefig('/scratch/tina/coeff39.png')
### <--- all files:

### 2 --- > this is for Gaussians
# nofcomponents = [2, 4, 8, 10, 12]
# for n in nofcomponents:
#     filename = 'GaussianHMM_' + str(n) + '_states/'
#     pr_npz = glob.glob('/scratch/tina/python_speech_features/coeff39/' + filename + '/*.npz')[0]
#     pr = np.load(pr_npz)
#     precision = pr['arr_0']
#     recall = pr['arr_1']
#     lab = os.path.basename(Path(filename)) + ' AUC: ' + str(round(auc(recall, precision), 4))
#     plt.step(recall, precision, label=lab)
#
# # plt.show()
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.legend(loc='upper right', fontsize='small')
# plt.savefig('/scratch/tina/Gaussians39.png')
### <---


### 3 --- > this is for GMMs coeff13
# fig = plt.figure(figsize=(10, 10))
# nofcomponents = [2, 4, 8, 10]
# nofcomponents2 = [2, 4, 8, 10, 12]
# aucs = []
# for n in nofcomponents:
#     for m in nofcomponents2:
#         if (n == 4 and m == 10) or (n == 4 and m == 12):
#             continue
#         filename = 'GMMHMM_' + str(n) + '_states_' + str(m) + '_mix/'
#         print(filename)
#         pr_npz = glob.glob('/scratch/tina/python_speech_features/coeff13/' + filename + '/*.npz')[0]
#         pr = np.load(pr_npz)
#         precision = pr['arr_0']
#         recall = pr['arr_1']
#         lab = os.path.basename(Path(filename)) + ' AUC: ' + str(round(auc(recall, precision), 4))
#         aucs.append(round(auc(recall, precision), 4))
#         plt.step(recall, precision, label=lab)
#
# # plt.show()
# print(sorted(aucs))
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.legend(loc='upper right', fontsize='small')
# plt.savefig('/scratch/tina/GMMs13.png')


#### coeff 39

fig = plt.figure(figsize=(20, 15))
nofcomponents = [2, 4, 8, 10, 12]
nofcomponents2 = [2, 4, 8, 10, 12]
aucs = []
for n in nofcomponents:
    for m in nofcomponents2:
        filename = 'GMMHMM_' + str(n) + '_states_' + str(m) + '_mix/'
        print(filename)
        pr_npz = glob.glob('/scratch/tina/python_speech_features/coeff39/' + filename + '/*.npz')
        # print(pr_npz)
        if not pr_npz:
            continue
        pr = np.load(pr_npz[0])
        precision = pr['arr_0']
        recall = pr['arr_1']
        lab = os.path.basename(Path(filename)) + ' AUC: ' + str(round(auc(recall, precision), 4))
        aucs.append(round(auc(recall, precision), 4))
        plt.step(recall, precision, label=lab)


for n in nofcomponents:
    filename = 'GaussianHMM_' + str(n) + '_states/'
    print(filename)
    pr_npz = glob.glob('/scratch/tina/python_speech_features/coeff39/' + filename + '/*.npz')
    # print(pr_npz)
    if not pr_npz:
        continue
    pr = np.load(pr_npz[0])
    precision = pr['arr_0']
    recall = pr['arr_1']
    lab = os.path.basename(Path(filename)) + ' AUC: ' + str(round(auc(recall, precision), 4))
    aucs.append(round(auc(recall, precision), 4))
    plt.step(recall, precision, label=lab)

# plt.show()
print(sorted(aucs))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='upper right', fontsize='small')
plt.savefig('/scratch/tina/coeff39.png')