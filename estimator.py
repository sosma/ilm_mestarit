from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression

def to01(row):
    if row['class4'] == "nonevent":
        return "nonevent"
    else:
        return "event"

df = pd.read_csv("npf_train.csv")
test = pd.read_csv("npf_test_hidden.csv")
find_numbers = [x for x in df.columns if x.endswith('.mean')]

data_nonscaled = df[find_numbers]
test_nonscaled = test[find_numbers]
data = data_nonscaled.rename(columns=lambda s: s[:-5])
data=(data_nonscaled-data_nonscaled.mean())/data_nonscaled.std()

test_data = test_nonscaled.rename(columns=lambda s: s[:-5])
test_data=(test_nonscaled-test_nonscaled.mean())/test_nonscaled.std()

data_joint = pd.concat([df['class4'],data],axis=1)

best_multi = [0, 0]
best_bin = [0, 0]
for i in range(1, 11):
    n_components = i

    pca = PCA(n_components=n_components).fit(pd.concat([test_data, data]))
    x_new = pca.transform(data)
    columns = ["pca"+str(x) for x in range(n_components)]
    pca_data = pd.DataFrame(x_new[:,0:n_components], columns=columns)
    joint_data = pd.concat([df['class4'],pca_data],axis=1)
    joint_data = joint_data.sample(frac=1).reset_index(drop=True)
    joint_data['class2'] = joint_data.apply(lambda row:  to01(row) , axis=1)

    test = joint_data.sample(frac = 0.5)
    train = joint_data.drop(test.index).reset_index(drop=True)
    # print(test)
    # print(train)
    X_train = train[[x for x in train.columns if x.startswith('pca')]]
    y_train = train['class4']
    X_test = test[[x for x in test.columns if x.startswith('pca')]]
    y_test = test['class4']
    clf = LogisticRegression(random_state=0).fit(X_train, y_train)
    acc_multi = (y_test == clf.predict(X_test)).value_counts().loc[True]
    if acc_multi >= best_multi[0]:
        best_multi = [acc_multi, n_components]


    X_train = train[[x for x in train.columns if x.startswith('pca')]]
    y_train = train['class2']
    X_test = test[[x for x in test.columns if x.startswith('pca')]]
    y_test = test['class2']
    clf = LogisticRegression(random_state=0).fit(X_train, y_train)
    acc_bin = (y_test == clf.predict(X_test)).value_counts().loc[True]
    if acc_bin >= best_bin[0]:
        best_bin = [acc_bin, n_components]

print("best multi: ", best_multi[0]/215, " correct, n_components = ", best_multi[1])
print("best binomial: ", best_bin[0]/215, " correct, n_components = ", best_bin[1])



"""
multiclass logistic regression
"""
pca_multiclass = PCA(n_components=best_multi[1]).fit(data)
x_new = pca.transform(data)
columns = ["pca"+str(x) for x in range(best_multi[1])]
pca_data = pd.DataFrame(x_new[:,0:best_multi[1]], columns=columns)
joint_data = pd.concat([df['class4'],pca_data],axis=1)
joint_data = joint_data.sample(frac=1).reset_index(drop=True)
joint_data['class2'] = joint_data.apply(lambda row:  to01(row) , axis=1)
X_train = joint_data[[x for x in joint_data.columns if x.startswith('pca')]]
y_train = joint_data['class4']

clf = LogisticRegression(random_state=0).fit(X_train, y_train)

x_test = pca.transform(test_data)
columns = ["pca"+str(x) for x in range(best_multi[1])]
pca_data = pd.DataFrame(x_test[:,0:best_multi[1]], columns=columns)

multi_predictions = clf.predict(pca_data)
multi_predictions_df = pd.DataFrame(multi_predictions, columns = ["class4"])


"""
binomial logistic regression
"""
pca_binomial = PCA(n_components=best_bin[1]).fit(data)
x_new = pca.transform(data)
columns = ["pca"+str(x) for x in range(best_bin[1])]
pca_data = pd.DataFrame(x_new[:,0:best_bin[1]], columns=columns)
joint_data = pd.concat([df['class4'],pca_data],axis=1)
joint_data = joint_data.sample(frac=1).reset_index(drop=True)
joint_data['class2'] = joint_data.apply(lambda row:  to01(row) , axis=1)

X_train = joint_data[[x for x in joint_data.columns if x.startswith('pca')]]
y_train = joint_data['class2']

clf = LogisticRegression(random_state=0).fit(X_train, y_train)

x_test = pca.transform(test_data)
columns = ["pca"+str(x) for x in range(best_bin[1])]
pca_data = pd.DataFrame(x_test[:,0:best_bin[1]], columns=columns)

binomial_predictions = clf.predict_proba(pca_data)
binomial_predictions_df = pd.DataFrame(binomial_predictions[:,0], columns = ["p"])

result = multi_predictions_df.join(binomial_predictions_df)
csv = result.to_csv(index=False)
with open("result.csv", "w") as f:
    f.write(str(best_bin[0]/215) + "\n")
    f.write(csv)
# print(result)
