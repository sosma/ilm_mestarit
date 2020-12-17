from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from itertools import chain, combinations
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, cross_validate, cross_val_score
import time

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
def PCA_editor(pca, indexes):
    for index in indexes:
        pca.components_[index] = np.zeros(100)
    return pca

def to01(row):
    if row['class4'] == "nonevent":
        return "nonevent"
    else:
        return "event"

df = pd.read_csv("npf_train.csv")
test = pd.read_csv("npf_test_hidden.csv")
find_numbers = [x for x in df.columns if x.endswith('.mean') or x.endswith('.std')]

data_nonscaled = df[find_numbers]
test_nonscaled = test[find_numbers]
data = data_nonscaled.rename(columns=lambda s: s[:-5])
data=(data_nonscaled-data_nonscaled.mean())/data_nonscaled.std()

test_data = test_nonscaled.rename(columns=lambda s: s[:-5])
test_data=(test_nonscaled-test_nonscaled.mean())/test_nonscaled.std()

data_joint = pd.concat([df['class4'],data],axis=1)

best_multi = [0, 0]
best_bin = [0, 0]
acc_multis = [0] * 50
acc_bins = [0] * 50
start_time = time.time()
n = 10
rounds=0
total_rounds = 2**n
for i in list(powerset(range(n))):
    rounds+=1
    if(not rounds%10):
        print(rounds/total_rounds)
    n_components = n
    pca_initial = PCA(n_components=n).fit(pd.concat([test_data, data]))
    pca = PCA_editor(pca_initial, i)
    x_new = pca.transform(data)
    columns = ["pca"+str(x) for x in range(n_components)]
    pca_data = pd.DataFrame(x_new[:,0:n_components], columns=columns)
    joint_data = pd.concat([df['class4'],pca_data],axis=1)
    joint_data = joint_data.sample(frac=1).reset_index(drop=True)
    joint_data['class2'] = joint_data.apply(lambda row:  to01(row) , axis=1)


    # TODO : cross validation here
    X = joint_data[[x for x in joint_data.columns if x.startswith('pca')]]
    y = joint_data['class4']
    n_splits = 10
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=69420)
    acc_multi = 0
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        clf = LogisticRegression(random_state=0).fit(X_train, y_train)
        acc_multi += (y_test == clf.predict(X_test)).value_counts().loc[True]
    if acc_multi >= best_multi[0]:
        best_multi = [acc_multi, i]

    y = joint_data['class2']
    acc_bin = 0
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        clf = LogisticRegression(random_state=0).fit(X_train, y_train)
        acc_bin += (y_test == clf.predict(X_test)).value_counts().loc[True]
    if acc_bin >= best_bin[0]:
        best_bin = [acc_bin, i]


print("--- %s seconds ---" % (time.time() - start_time))

print("best multi: ", best_multi[0]/430, " left out components: ", best_multi[1])
print("best binomial: ", best_bin[0]/430, " left out components: ", best_bin[1])



"""
multiclass logistic regression
"""
pca_multiclass = PCA(n_components=n).fit(data)
pca = PCA_editor(pca_multiclass, best_multi[1])
x_new = pca_multiclass.transform(data)
columns = ["pca"+str(x) for x in range(n)]
pca_data = pd.DataFrame(x_new[:,0:n], columns=columns)
joint_data = pd.concat([df['class4'],pca_data],axis=1)
joint_data = joint_data.sample(frac=1).reset_index(drop=True)
joint_data['class2'] = joint_data.apply(lambda row:  to01(row) , axis=1)
X_train = joint_data[[x for x in joint_data.columns if x.startswith('pca')]]
y_train = joint_data['class4']

clf = LogisticRegression(random_state=0).fit(X_train, y_train)

x_test = pca.transform(test_data)
columns = ["pca"+str(x) for x in range(n)]
pca_data = pd.DataFrame(x_test[:,0:n], columns=columns)

multi_predictions = clf.predict(pca_data)
multi_predictions_df = pd.DataFrame(multi_predictions, columns = ["class4"])


"""
binomial logistic regression
"""
pca_binomial = PCA(n_components=n).fit(data)
pca = PCA_editor(pca_binomial, best_bin[1])
x_new = pca.transform(data)
columns = ["pca"+str(x) for x in range(n)]
pca_data = pd.DataFrame(x_new[:,0:n], columns=columns)
joint_data = pd.concat([df['class4'],pca_data],axis=1)
joint_data = joint_data.sample(frac=1).reset_index(drop=True)
joint_data['class2'] = joint_data.apply(lambda row:  to01(row) , axis=1)

X_train = joint_data[[x for x in joint_data.columns if x.startswith('pca')]]
y_train = joint_data['class2']

clf = LogisticRegression(random_state=0).fit(X_train, y_train)

x_test = pca.transform(test_data)
columns = ["pca"+str(x) for x in range(n)]
pca_data = pd.DataFrame(x_test[:,0:n], columns=columns)

binomial_predictions = clf.predict_proba(pca_data)
binomial_predictions_df = pd.DataFrame(binomial_predictions[:,0], columns = ["p"])

result = multi_predictions_df.join(binomial_predictions_df)
csv = result.to_csv(index=False)
with open("result.csv", "w") as f:
    f.write(str(best_bin[0]/430) + "\n")
    f.write(csv)
# print(result)

# Plotting accuracies with different number of components used
#
# plt.plot(acc_multis)
# plt.title('Multiclass')
# plt.xlabel('Number of components')
# plt.ylabel('Accuracy')
# plt.show()
#
# plt.plot(acc_bins)
# plt.title('Binary')
# plt.xlabel('Number of components')
# plt.ylabel('Accuracy')
# plt.show()
