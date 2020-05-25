import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score
from time import time

category_to_id = {'tech': 0, 'business': 1, 'sport': 2, 'entertainment': 3, 'politics': 4}
id_to_category = {}
for key in category_to_id:
	id_to_category[category_to_id[key]] = key


data = pd.read_csv('./data/bbc-text.csv')

data['category_id'] = [category_to_id[x] for x in data['category']]

X_train, X_test, y_train, y_test = train_test_split(data['text'], data['category_id'], random_state = 3)

cv = CountVectorizer(strip_accents='ascii', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', lowercase=True, stop_words='english')

l = len(X_train)
X_cv = cv.fit_transform([*X_train, *X_test])
X_train_cv = X_cv[:l]
X_test_cv = X_cv[l:]

naive_bayes = MultinomialNB()
naive_bayes.fit(X_train_cv, y_train)
predictions = naive_bayes.predict(X_test_cv)

now = time()

accuracy = accuracy_score(y_test, predictions)

train = pd.DataFrame({"text": X_train, "category": y_train})
train.to_csv('./log/train_time_' + str(now) + '_accuracy_' + str(accuracy) + '.csv', index=True)

result = pd.DataFrame({"text": X_test, "category": y_test, "prediction": predictions})
result.to_csv('./log/test_time_' + str(now) + '_accuracy_' + str(accuracy) + '.csv', index=True)

print('Accuracy score: ', accuracy_score(y_test, predictions))