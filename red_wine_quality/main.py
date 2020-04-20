from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import graphviz
import pandas as pd
from time import time
from copy import copy, deepcopy

# X = [[15, 1, 0], [20, 3, 1], [25, 2, 0], [30, 4, 0], [35, 2, 1], [25, 4, 0], [15, 2, 1], [20, 3, 1]]
# y = [0, 1, 0, 0, 1, 0, 1, 1]
# X_test = [[10, 2, 0], [20, 1, 1], [30, 3, 0], [40, 2, 1], [15, 1, 1]]
# y_true = [1, 0, 1, 1, 0]

data = pd.read_csv('./data/winequality-red.csv')
feature_names = list(data.columns)[:-1]
class_names = ['not good', 'good']

y = [int(x >= 6) for x in data['quality']]
X = deepcopy(data.drop(columns = ['quality']))

X, X_test, y, y_true = train_test_split(X, y, random_state = 7)

clf = tree.DecisionTreeClassifier()
start = time()
clf = clf.fit(X, y)
end = time()


y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_true, y_pred)
print(y_pred)
print('Accuracy: ' + str(accuracy))

tree.plot_tree(clf)

dot_data = tree.export_graphviz(clf, out_file=None,
	feature_names=feature_names,
	class_names=class_names,
	filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render('./model/tree_' + str(start))

train = pd.DataFrame({**X, "quality": y})
train.to_csv('./log/train_time_' + str(start) + '_accuracy_' + str(accuracy) + '.csv', index=True)

result = pd.DataFrame({**X_test, "true quality": y_true, "predicted quality": y_pred})
result.to_csv('./log/test_time_' + str(start) + '_accuracy_' + str(accuracy) + '.csv', index=True)