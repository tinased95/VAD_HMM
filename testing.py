from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

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


x = np.load('x.npy')
y = np.load('y.npy')
patient_ids = np.load('patient_ids.npy')

x.tolist()
y.tolist()

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, shuffle=False)

range_list_1 = get_windows(y_test, 1)
range_list_0 = get_windows(y_test, 0)

y_new = sorted(range_list_0 + range_list_1) # windowed y labels and ranges


x_test_new = []
y_test_new = []
for s_e_l in y_new:
    lists = []
    for se in range(s_e_l[0], s_e_l[1]):
        lists.append(X_test[se].flatten())
    lists = np.asarray(lists)
#     print(lists.shape)
    x_test_new.append(lists)
    y_test_new.append(s_e_l[2])
x_test_new = np.asarray(x_test_new)
y_test_new = np.asarray(y_test_new)


# print((y_new))
print((x_test_new.shape))
print((y_test_new.shape))


def prediction(test_data, trained):
    # predict list of test
    predict_label = []
    predict_probs = []
    predict_max = []
    if type(test_data) == type([]):
        for test in tqdm(test_data):
            scores = []
            for node in trained.keys():
                scores.append(trained[node].score(test))
            predict_label.append(scores.index(max(scores)))
            predict_probs.append(scores[0] - scores[1])
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


with open("test.pkl", "rb") as file:
    learned_hmm = pickle.load(file)

y_pred, y_probs, y_max = prediction(list(x_test_new), learned_hmm)

report(y_test_new, y_pred, show_cm=True)
