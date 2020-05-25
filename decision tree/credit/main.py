from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import graphviz
import pandas as pd
from time import time
from sklearn import preprocessing

def read_data(file='./data/credit_train.csv'):
	raw_data = pd.read_csv(file)
	raw_data = raw_data.drop(columns = ['Loan ID', 'Customer ID'])
	if 'Loan Status' in raw_data.columns:
		raw_data = raw_data.drop(columns = ['Loan Status'])
	data = dict()
	dict_data = dict(raw_data)
	le = preprocessing.LabelEncoder()
	for column in dict_data:
		transform_list = [str(x) for x in dict_data[column]]
		fit_list = transform_list
		le.fit(fit_list)
		data[column] = le.transform(transform_list)
	data = pd.DataFrame(data)

	feature_names = list(data.columns)[:-1]
	y = [x for x in data['Bankruptcies']]
	X = data.drop(columns = ['Bankruptcies'])
	return X, y, feature_names

X, y, feature_names = read_data('./data/credit_train.csv')
X_test, y_true, _ = read_data('./data/credit_test.csv')

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
	filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render('./model/' + str(start) + '_accuracy_' + str(accuracy) + '_model')

train = pd.DataFrame({**X, "class": y})
train.to_csv('./log/' + str(start) + '_accuracy_' + str(accuracy) + '_train.csv', index=True)

result = pd.DataFrame({**X_test, "true class": y_true, "predicted class": y_pred})
result.to_csv('./log/' + str(start) + '_accuracy_' + str(accuracy) + '_test.csv', index=True)