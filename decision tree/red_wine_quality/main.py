from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import graphviz
import pandas as pd
from time import time

data = pd.read_csv('./data/winequality-red.csv')
feature_names = list(data.columns)[:-1]
class_names = ['not good', 'good']

y = [int(x >= 6) for x in data['quality']]
X = data.drop(columns = ['quality'])

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