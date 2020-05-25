from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import graphviz
import pandas as pd
from time import time
from sklearn import preprocessing

raw_data = pd.read_csv('./data/mushrooms.csv')
data = dict()
dict_data = dict(raw_data)
le = preprocessing.LabelEncoder()
for column in dict_data:
	fit_list = dict_data[column]
	if (column == 'class'):
		fit_list = ['p', 'e']
	le.fit(fit_list)
	data[column] = le.transform(dict_data[column])
data = pd.DataFrame(data)

feature_names = list(data.columns)[:-1]
class_names = ['p', 'e']

y = [x for x in data['class']]
X = data.drop(columns = ['class'])

X, X_test, y, y_true = train_test_split(X, y, random_state = 7)

clf = tree.DecisionTreeClassifier()
start = time()
clf = clf.fit(X, y)
end = time()


y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_true, y_pred)
print('Prediction: ', y_pred)
print('Accuracy: ' + str(accuracy))

tree.plot_tree(clf)

dot_data = tree.export_graphviz(clf, out_file=None,
	feature_names=feature_names,
	class_names=class_names,
	filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render('./model/tree_' + str(start))

train = pd.DataFrame({**X, "class": y})
train.to_csv('./log/train_time_' + str(start) + '_accuracy_' + str(accuracy) + '.csv', index=True)

result = pd.DataFrame({**X_test, "true class": y_true, "predicted class": y_pred})
result.to_csv('./log/test_time_' + str(start) + '_accuracy_' + str(accuracy) + '.csv', index=True)