import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier


data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
df.head()

# test_size: what proportion of original data is used for test set
X_train, X_test, Y_train, Y_test = train_test_split(df[data.feature_names],
                                                    df['target'],
                                                    train_size = .75,
                                                    random_state=0)

clf = DecisionTreeClassifier(max_depth = 2, 
                             random_state = 0)

clf.fit(X_train, Y_train)
DecisionTreeClassifier(max_depth=2, random_state=0)
# Returns a NumPy Array
# Predict for One Observation (image)
clf.predict(X_test.iloc[0].values.reshape(1, -1))
clf.predict(X_test[0:10])
score = clf.score(X_test, Y_test)
# List of values to try for max_depth:
max_depth_range = list(range(1, 6))

# List to store the average RMSE for each value of max_depth:
accuracy = []

for depth in max_depth_range:
    
    clf = DecisionTreeClassifier(max_depth = depth, 
                             random_state = 0)
    clf.fit(X_train, Y_train)

    score = clf.score(X_test, Y_test)
    accuracy.append(score)

fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10,7));

ax.plot(max_depth_range,
        accuracy,
        lw=2,
        color='k')

ax.set_xlim([1, 5])
ax.set_ylim([.50, 1.00])
ax.grid(True,
        axis = 'both',
        zorder = 0,
        linestyle = ':',
        color = 'k')

yticks = ax.get_yticks()

y_ticklist = []
for tick in yticks:
    y_ticklist.append(str(tick).ljust(4, '0')[0:4])
ax.set_yticklabels(y_ticklist)
ax.tick_params(labelsize = 18)
ax.set_xticks([1,2,3,4,5])
ax.set_xlabel('max_depth', fontsize = 24)
ax.set_ylabel('Accuracy', fontsize = 24)
fig.tight_layout()
fig.savefig('images/max_depth_vs_entropy.png', dpi = 300)